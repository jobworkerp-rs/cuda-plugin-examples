// from https://github.com/huggingface/candle/blob/main/candle-examples/examples/whisper/main.rs
// TODO remove old candle codes

pub mod audio;
pub mod model;
pub mod multilingual;
pub mod segment;

// https://github.com/openai/whisper/blob/main/whisper/model.py/rgs
// TODO:
// - Batch size greater than 1.
// - More token filters (SuppressBlanks, ApplyTimestampRules).

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

use self::segment::{DecodingResult, Segment};
use crate::audio::trans::AudioTensorProcessor;
use anyhow::{anyhow, Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::whisper::{
    EOT_TOKEN, NO_SPEECH_TOKENS, NO_TIMESTAMPS_TOKEN, SOT_TOKEN, TEMPERATURES, TRANSCRIBE_TOKEN,
    TRANSLATE_TOKEN,
};
use candle_wrapper::{models::token_id, whisper::WhisperWrapper};
use clap::ValueEnum;
use rand::{distributions::Distribution, SeedableRng};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

// Audio parameters.
pub const CHUNK_LENGTH: usize = candle_transformers::models::whisper::CHUNK_LENGTH; // seconds
const N_SAMPLES: usize = CHUNK_LENGTH * AudioTensorProcessor::SAMPLE_RATE; // 480000 samples in a 30-second chunk
const N_FRAMES: usize = N_SAMPLES / AudioTensorProcessor::HOP_LENGTH; // 3000 frames in a mel spectrogram input

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize, PartialEq)]
pub enum Task {
    Transcribe,
    Translate,
}

pub struct Decoder {
    model: WhisperWrapper,
    rng: rand::rngs::StdRng,
    task: Task,
    timestamps: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    const NO_SPEECH_THRESHOLD: f64 = 0.6;
    const LOGPROB_THRESHOLD: f64 = -1.0;
    const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

    pub fn new(
        model: WhisperWrapper,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Task,
        timestamps: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, EOT_TOKEN)?;
        let no_speech_token = NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        tracing::debug!("audio features: {:?}", audio_features.dims());
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            Task::Transcribe => tokens.push(self.transcribe_token),
            Task::Translate => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > Self::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < Self::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > Self::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    tracing::debug!("Error running at {t}: {err}")
                }
            }
        }
        Err(anyhow!("should not happen (failed decoding: {:?}", segment))
    }

    pub fn run(&mut self, processor: &mut AudioTensorProcessor) -> Result<Vec<Segment>> {
        let mut segments = vec![];
        let sample_rate = AudioTensorProcessor::SAMPLE_RATE;

        // set first tensor
        let mut buffer = Some(processor.get_current_mel_tensor()?);
        // content loop (until finish decoding all audio)
        let start_time = std::time::Instant::now();
        let mut i = 0;
        while let Some(tensor) = buffer {
            let seek = 0;
            let (_, _, content_frames) = tensor.dims3()?;

            // 1 tensor loop (1buffer size: usually 30 sec)
            // (shift audio buffer as same as 1 buffer size, so  iterate current frames first buffer only (not whole))
            // while seek < content_frames {
            let time_offset = (seek * AudioTensorProcessor::HOP_LENGTH) as f64 / sample_rate as f64;
            let segment_size = usize::min(content_frames - seek, N_FRAMES);
            let mel_segment = tensor.narrow(2, seek, segment_size)?;
            let segment_duration =
                (segment_size * AudioTensorProcessor::HOP_LENGTH) as f64 / sample_rate as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;

            if dr.no_speech_prob > Self::NO_SPEECH_THRESHOLD
                && dr.avg_logprob < Self::LOGPROB_THRESHOLD
            {
                tracing::debug!("no speech detected, skipping {seek} {dr:?}");
            } else {
                let segment = Segment {
                    start: (CHUNK_LENGTH * i) as f64 + time_offset,
                    duration: segment_duration,
                    dr,
                };
                if self.timestamps {
                    self.print_relative_timestamp(&segment)?;
                }
                tracing::debug!(
                    "[{i}] {seek}/{content_frames}: {segment:?}, in {:?}",
                    start_time.elapsed()
                );
                segments.push(segment);
            }
            // seek += segment_size;
            // }

            // read and set next tensor buffer
            buffer = processor.load_next_mel_tensor()?;
            i += 1;
        }
        Ok(segments)
    }
    fn print_relative_timestamp(&self, segment: &Segment) -> Result<()> {
        tracing::info!(
            "{:.1}s -- {:.1}s",
            segment.start,
            segment.start + segment.duration,
        );
        let mut tokens_to_decode = vec![];
        let mut prev_timestamp_s = 0f32;
        for &token in segment.dr.tokens.iter() {
            if token == self.sot_token || token == self.eot_token {
                continue;
            }
            // The no_timestamp_token is the last before the timestamp ones.
            if token > self.no_timestamps_token {
                let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                if !tokens_to_decode.is_empty() {
                    let text = self
                        .tokenizer
                        .decode(&tokens_to_decode, true)
                        .map_err(E::msg)?;
                    tracing::info!("  {:.1}s-{:.1}s: {}", prev_timestamp_s, timestamp_s, text);
                    tokens_to_decode.clear()
                }
                prev_timestamp_s = timestamp_s;
            } else {
                tokens_to_decode.push(token)
            }
        }
        if !tokens_to_decode.is_empty() {
            let text = self
                .tokenizer
                .decode(&tokens_to_decode, true)
                .map_err(E::msg)?;
            if !text.is_empty() {
                tracing::info!("  {:.1}s-...: {}", prev_timestamp_s, text);
            }
            tokens_to_decode.clear()
        }
        Ok(())
    }
}
