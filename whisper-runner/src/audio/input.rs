use std::fs::File;
use std::path::Path;

use symphonia::core::audio::{AudioBuffer, Channels, Signal, SignalSpec};
use symphonia::core::codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::conv::FromSample;
use symphonia::core::errors::Error;
use symphonia::core::errors::Result;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub struct AudioNormalizedDecoder {
    format: Box<dyn FormatReader>,
    track_id: u32,
    decoder: Box<dyn Decoder>,
    // empty until first decode
    pub spec: SignalSpec,
    // empty until first decode
    decode_size: usize,
    sample_buf: Option<AudioBuffer<f32>>,
    pub total_frames: Option<u64>,
    pub current_frame: u64,
    pub buf_duration_sec: u64,
}

impl AudioNormalizedDecoder {
    // new setup from input file string
    pub fn new(file: String, buf_duration_sec: u64) -> Result<Self> {
        let path = std::path::PathBuf::from(file);
        let ext = path.extension().and_then(|ex| ex.to_str());
        // Open the media source.
        let src = std::fs::File::open(path.as_path().to_str().unwrap())?;
        // Create the media source stream.
        let mss = MediaSourceStream::new(Box::new(src), Default::default());
        // Create a probe hint using the file's extension. [Optional]
        let mut hint = Hint::new();
        ext.map(|ex| hint.with_extension(ex));
        Self::setup(mss, hint, buf_duration_sec)
    }
    // Create the media source stream from data.
    pub fn new_by_data(
        data: Box<dyn MediaSource>,
        ext: Option<&String>,
        buf_duration_sec: u64,
    ) -> Result<Self> {
        let mss = MediaSourceStream::new(data, Default::default());
        // Create a probe hint using the file's extension. [Optional]
        let mut hint = Hint::new();
        ext.map(|ex| hint.with_extension(ex));
        Self::setup(mss, hint, buf_duration_sec)
    }
    fn setup(mss: MediaSourceStream, hint: Hint, buf_duration_sec: u64) -> Result<Self> {
        // Use the default options for metadata and format readers.
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        // Probe the media source.
        let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;

        // Get the instantiated format reader.
        let format = probed.format;

        // Find the first audio track with a known (decodeable) codec.
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or(symphonia::core::errors::Error::Unsupported(
                "no supported audio tracks",
            ))?;
        let track_id = track.id;

        // Use the default options for the decoder.
        let dec_opts: DecoderOptions = Default::default();

        let total = track.codec_params.n_frames;
        // Create a decoder for the track.
        let decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

        Ok(Self {
            format,
            track_id,
            decoder,
            spec: SignalSpec {
                channels: Channels::FRONT_LEFT,
                rate: 0,
            },
            decode_size: 0,
            sample_buf: None,
            total_frames: total,
            current_frame: 0,
            buf_duration_sec,
        })
    }
    #[inline]
    pub fn is_filled(&self) -> bool {
        if let Some(sbuf) = self.sample_buf.as_ref() {
            // filled: cannot store next decode
            sbuf.frames() + self.decode_size > sbuf.capacity()
        } else {
            false
        }
    }
    pub fn clear_buffer(&mut self) {
        if let Some(sbuf) = self.sample_buf.as_mut() {
            sbuf.clear(); // reset buffer write position to 0
        }
    }
    pub fn shift_buffer(&mut self, duration_sec: usize) {
        if let Some(sbuf) = self.sample_buf.as_mut() {
            sbuf.shift(duration_sec * self.spec.rate as usize);
        }
    }
    pub fn get_buffer(&mut self) -> Result<&AudioBuffer<f32>> {
        if self.sample_buf.is_none() {
            self.decode_buffered_fill_next(None)?;
        }
        if let Some(buf) = self.sample_buf.as_ref() {
            Ok(buf)
        } else {
            Err(Error::Unsupported("no buffer: failed to decode input"))
        }
    }
    // decode until filling buf_duration
    #[inline]
    pub fn decode_buffered_fill_next(
        &mut self,
        shift: Option<usize>,
    ) -> Result<Option<&AudioBuffer<f32>>> {
        if let Some(s) = shift {
            self.shift_buffer(s);
        } else {
            // XXX: clear buffer when shift is not specified
            self.clear_buffer();
            assert!(!self.is_filled());
        }
        while let Ok(buf_opt) = self.decode_buffered_next(self.buf_duration_sec) {
            if buf_opt.is_none() {
                if let Some(sbuf) = self.sample_buf.as_ref() {
                    if sbuf.frames() > 0 {
                        return Ok(self.sample_buf.as_ref());
                    }
                }
                return Ok(None);
            }
            if self.is_filled() {
                break;
            }
        }
        Ok(self.sample_buf.as_ref())
    }
    #[inline]
    fn decode_buffered_next(&mut self, buf_duration_sec: u64) -> Result<Option<&AudioBuffer<f32>>> {
        let decoded = self.decode_next().map_err(|e| {
            tracing::warn!("decode error:{:?}", e);
            e
        })?;
        // single channel spec
        let one_spec = SignalSpec {
            rate: self.spec.rate,
            channels: Channels::FRONT_LEFT,
        };
        if let Some(mut dec) = decoded {
            // first time decode (init sample buffer)
            if self.sample_buf.is_none() {
                // Create the f32 mono-channel sample buffer.
                // create buffer that has the enough capacity of buf_duration seconds (extend frame size (times decoded frames()))
                let total_buf_samples = buf_duration_sec * self.spec.rate as u64;
                let real_total_buf_samples =
                    (total_buf_samples / dec.frames() as u64 + 1) * dec.frames() as u64;
                tracing::debug!(
                    "unit frame: {}, total_buf_samples: {}, real_total_buf_samples: {}",
                    dec.frames(),
                    total_buf_samples,
                    real_total_buf_samples
                );
                let sbuf = AudioBuffer::<f32>::new(real_total_buf_samples, one_spec);
                self.sample_buf = Some(sbuf);
            }

            if let Some(sbuf) = self.sample_buf.as_mut() {
                // check if buffer will be full with input of next decode frames
                if sbuf.capacity() - sbuf.frames() < self.decode_size {
                    sbuf.shift(dec.frames());
                }
                // extend frame size
                // sbuf.render_reserved(Some(dec.frames() as usize));
                match self.spec.channels.count() {
                    1 => {
                        sbuf.render(Some(dec.frames()), |planes, _frame_pos| {
                            let mut iter = planes.planes()[0].iter_mut();
                            for sample in dec.chan(0) {
                                if let Some(s) = iter.next() {
                                    *s = *sample; //f32::from_sample(*sample);
                                }
                            }
                            Ok(())
                        })?;
                        // The samples may now be access via the `samples()` function.
                        self.current_frame += dec.frames() as u64;
                        tracing::trace!(
                            "1Decoded samples in {}/{}, buf({}/{:?})",
                            sbuf.frames(),
                            sbuf.capacity(),
                            self.current_frame,
                            self.total_frames,
                        );

                        Ok(self.sample_buf.as_ref())
                    }
                    2 => {
                        sbuf.render(Some(dec.frames()), |planes, _frame_pos| {
                            // mux to mono from dec.chan_pair_mut() to planes
                            let (lbuf, rbuf) = dec.chan_pair_mut(0, 1);
                            let mut iter = planes.planes()[0].iter_mut();
                            for (l, r) in lbuf.iter_mut().zip(rbuf) {
                                if let Some(s) = iter.next() {
                                    *s = f32::from_sample((*l + *r) / 2.0);
                                } else {
                                    tracing::warn!("buffer full");
                                }
                            }
                            Ok(())
                        })?;
                        self.current_frame += dec.frames() as u64;
                        tracing::trace!(
                            "2Decoded samples in {}/{}, buf({}/{:?})",
                            sbuf.frames(),
                            sbuf.capacity(),
                            self.current_frame,
                            self.total_frames,
                        );
                        Ok(self.sample_buf.as_ref())
                    }
                    c => {
                        tracing::error!("unsupported channel count: {}", c);
                        Err(Error::Unsupported("unsupported channel count"))
                    }
                }
            } else {
                tracing::warn!("sample buffer not initialized: illegal state");
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    #[inline]
    fn decode_next(&mut self) -> Result<Option<AudioBuffer<f32>>> {
        loop {
            // Get the next packet from the media format.
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(Error::ResetRequired) => {
                    // The track list has been changed. Re-examine it and create a new set of decoders,
                    // then restart the decode loop. This is an advanced feature and it is not
                    // unreasonable to consider this "the end." As of v0.5.0, the only usage of this is
                    // for chained OGG physical streams.
                    return Err(Error::ResetRequired); // unimplemented
                }
                Err(Error::IoError(e)) => {
                    // when next packet failed to read due to an IO error, consider as end of stream.
                    tracing::debug!("IO error: {:?}", e);
                    return Ok(None);
                }
                Err(err) => {
                    // A unrecoverable error occurred, halt decoding with error.
                    // panic!("{}", err);
                    tracing::warn!("next packet: {:?}", err);
                    return Err(err);
                }
            };

            // If the packet does not belong to the selected track, skip over it.
            if packet.track_id() != self.track_id {
                continue;
            }

            // // Consume any new metadata that has been read since the last packet.
            // while !format.metadata().is_latest() {
            //     // Pop the old head of the metadata queue.
            //     let m = format.metadata().pop();
            //     tracing::info!("Metadata: {:?}", m);
            //     // Consume the new metadata at the head of the metadata queue.
            // }

            // Decode the packet into audio samples.
            match self.decoder.decode(&packet) {
                Ok(decoded) => {
                    if self.decode_size == 0 {
                        self.decode_size = decoded.capacity();
                        self.spec = *decoded.spec();
                        decoded.spec().channels.count();
                    } else if self.decode_size != decoded.capacity() {
                        tracing::error!(
                            "capacity mismatch! {} != {}",
                            self.decode_size,
                            decoded.capacity()
                        );
                        return Err(Error::Unsupported("capacity mismatch"));
                    } else if &self.spec != decoded.spec() {
                        tracing::error!("spec mismatch! {:?} != {:?}", self.spec, decoded.spec());
                        return Err(Error::Unsupported("spec mismatch"));
                    }
                    tracing::trace!("decoded: {:?}", decoded.frames());
                    let mut abuf =
                        AudioBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                    // copy from decoder buffer
                    decoded.convert::<f32>(&mut abuf);

                    return Ok(Some(abuf));
                }
                Err(Error::IoError(e)) => {
                    // The packet failed to decode due to an IO error, skip the packet.
                    // continue;
                    tracing::warn!("IO error: {:?}", e);
                    return Ok(None);
                }
                Err(Error::DecodeError(_)) => {
                    // The packet failed to decode due to invalid data, skip the packet.
                    continue;
                }
                Err(err) => {
                    // An unrecoverable error occurred, halt decoding.
                    tracing::warn!("{}", err);
                    // panic!("{}", err);
                    return Ok(None);
                }
            }
        }
    }

    // for test (unused usually)
    fn _write_wav(&self, out_filename: &String) -> Result<()> {
        let header = wav::Header::new(wav::header::WAV_FORMAT_IEEE_FLOAT, 1, self.spec.rate, 32);
        let buf = self.sample_buf.as_ref().unwrap().chan(0);
        let mut out_file = File::create(Path::new(out_filename))?;
        wav::write(
            header,
            &wav::BitDepth::ThirtyTwoFloat(buf.to_vec()),
            &mut out_file,
        )?;
        Ok(())
    }
}
