use crate::audio::input::AudioNormalizedDecoder;
use crate::runner::whisper::audio::{self, Float};
use anyhow::{anyhow, Context, Result};
use candle_core::{Device, Tensor};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::io::MediaSource;

// Audio parameters.
static MEL_FILTERS80: once_cell::sync::Lazy<Vec<f32>> = once_cell::sync::Lazy::new(|| {
    let mel_bytes = include_bytes!("../../resources/melfilters.bytes");
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    mel_filters
});

static MEL_FILTERS128: once_cell::sync::Lazy<Vec<f32>> = once_cell::sync::Lazy::new(|| {
    let mel_bytes = include_bytes!("../../resources/melfilters128.bytes");
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    mel_filters
});

pub struct AudioTensorProcessor {
    pub audio_normalized_decoder: AudioNormalizedDecoder,
    pub device: Device,
    pub original_sample_rate: usize,
    pub n_mels: usize,
    mel_filters: &'static [f32],
    pub resampler: Option<MyResampler>,
    pub chunk_length: usize,
}
impl AudioTensorProcessor {
    const N_FFT: usize = 400;
    // const N_MELS: usize = 80;
    pub const SAMPLE_RATE: usize = 16000;
    pub const HOP_LENGTH: usize = 160;

    pub fn new_by_data(
        data: Box<dyn MediaSource>,
        ext: Option<&String>,
        buf_sec: u64,
        n_mels: usize,
        n_tracks: usize,
    ) -> Result<Self> {
        let audio = AudioNormalizedDecoder::new_by_data(data, ext, buf_sec, n_tracks)?;
        Self::new(audio, buf_sec as usize, n_mels)
    }

    pub fn new_by_input(
        input: String,
        buf_sec: u64,
        n_mels: usize,
        n_tracks: usize,
    ) -> Result<Self> {
        let audio = AudioNormalizedDecoder::new(input, buf_sec, n_tracks)?;
        Self::new(audio, buf_sec as usize, n_mels)
    }
    pub fn new(audio: AudioNormalizedDecoder, chunk_length: usize, n_mels: usize) -> Result<Self> {
        let device = Device::cuda_if_available(0).context("no cuda device")?;
        let sample_rate = audio.spec.rate as usize; // 0 if not decoded yet
        let mel_filters = if n_mels == 128 {
            MEL_FILTERS128.as_slice() // v3
        } else {
            MEL_FILTERS80.as_slice()
        };
        Ok(Self {
            audio_normalized_decoder: audio,
            device,
            original_sample_rate: sample_rate,
            n_mels,
            mel_filters,
            resampler: None,
            chunk_length,
        })
    }
    // initialize buffer and resampler
    pub fn get_current_mel_tensor(&mut self) -> Result<Tensor> {
        let sbuf = self.audio_normalized_decoder.get_buffer()?;
        // let mel = Self::pcm_to_mel(&sbuf.planes().planes()[0].to_vec(), MEL_FILTERS.as_slice())?;
        let mut resmpl = MyResampler::new(sbuf.spec().rate as usize, Self::SAMPLE_RATE);
        let buf = resmpl.resample(sbuf.planes().planes()[0])?;
        // for test to hear the resampled audio
        // self.write_wav(&"test.wav".to_string(), &buf.to_vec())?;
        let mel = Self::pcm_to_mel(&buf, self.mel_filters, self.chunk_length, self.n_mels)?;
        let mel_len = mel.len();
        let mel = Tensor::from_vec(mel, (1, self.n_mels, mel_len / self.n_mels), &self.device)?;
        self.original_sample_rate = self.audio_normalized_decoder.spec.rate as usize;
        self.resampler = Some(resmpl);

        tracing::debug!("loaded mel: {:?}", mel.dims());
        Ok(mel)
    }
    pub fn load_next_mel_tensor(&mut self) -> Result<Option<Tensor>> {
        if let Some(sbuf) = self.get_resampled_buf()? {
            // let mel =
            //     self.load_mel_tensor_from(sbuf.planes().planes()[0].to_vec(), &self.device)?;
            // resample to 16k
            let mel = Self::pcm_to_mel(&sbuf, self.mel_filters, self.chunk_length, self.n_mels)?;
            let mel_len = mel.len();
            let mel = Tensor::from_vec(mel, (1, self.n_mels, mel_len / self.n_mels), &self.device)?;
            tracing::debug!("loaded mel: {:?}", mel.dims());

            Ok(Some(mel))
        } else {
            Ok(None)
        }
    }
    fn get_resampled_buf(&mut self) -> Result<Option<Vec<f32>>> {
        let resampler = self.resampler.as_mut().unwrap();
        if let Some(sbuf) = self
            .audio_normalized_decoder
            .decode_buffered_fill_next(Some(self.chunk_length))?
        {
            let out = resampler.resample(sbuf.planes().planes()[0])?;
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }
    fn pcm_to_mel<T: Float + std::fmt::Display>(
        samples: &[T],
        filters: &[T],
        chunk_length: usize,
        n_mels: usize,
    ) -> anyhow::Result<Vec<T>> {
        let mel = audio::log_mel_spectrogram_(
            samples,
            filters,
            Self::N_FFT,
            Self::HOP_LENGTH,
            n_mels,
            false,
            chunk_length,
        );
        Ok(mel)
    }
}

pub struct MyResampler {
    resampler: SincFixedIn<f32>,
    output_buffer: Vec<Vec<f32>>,
    rate: f32,
}
impl MyResampler {
    pub fn new(from_sample_rate: usize, to_sample_rate: usize) -> Self {
        let sinc_len = 128;
        let window = WindowFunction::Blackman2;

        let f_cutoff = rubato::calculate_cutoff(sinc_len, window);
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };

        let resampler = SincFixedIn::<f32>::new(
            to_sample_rate as f64 / from_sample_rate as f64,
            8.0,
            params,
            1024,
            1,
        )
        .unwrap();
        let waves_out = vec![vec![0f32; resampler.output_frames_max()]];
        Self {
            resampler,
            output_buffer: waves_out,
            rate: to_sample_rate as f32 / from_sample_rate as f32,
        }
    }

    // ref. https://github.com/HEnquist/rubato/blob/master/examples/process_f64.rs#L188
    pub fn resample(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let waves_in = [samples.to_vec()];
        let outbuffer = self.output_buffer.as_mut();
        let mut indata_slices: Vec<&[f32]> = waves_in.iter().map(|v| &v[..]).collect();
        let mut input_frames_next = self.resampler.input_frames_next();
        // Create buffer for storing output
        let mut outdata = vec![Vec::with_capacity(
            (samples.len() as f32 * self.rate) as usize,
        )];

        while indata_slices[0].len() >= input_frames_next {
            let (nbr_in, nbr_out) = self
                .resampler
                .process_into_buffer(&indata_slices, outbuffer, None)
                .unwrap();
            for chan in indata_slices.iter_mut() {
                *chan = &chan[nbr_in..];
            }
            Self::append_frames(&mut outdata, outbuffer, nbr_out);
            input_frames_next = self.resampler.input_frames_next();
        }

        // Process a partial chunk with the last frames.
        if !indata_slices[0].is_empty() {
            let (_nbr_in, nbr_out) = self
                .resampler
                .process_partial_into_buffer(Some(&indata_slices), outbuffer, None)
                .unwrap();
            Self::append_frames(&mut outdata, outbuffer, nbr_out);
        }

        // let mut waves_out = resampler.process(&waves_in, None)?;
        //        tracing::debug!("resampled waves: {:?}", waves_out.len());
        self.resampler.reset();
        outdata.pop().ok_or(anyhow!("no resampled waves"))
    }

    fn append_frames(buffers: &mut [Vec<f32>], additional: &[Vec<f32>], nbr_frames: usize) {
        buffers
            .iter_mut()
            .zip(additional.iter())
            .for_each(|(b, a)| b.extend_from_slice(&a[..nbr_frames]));
    }
}
