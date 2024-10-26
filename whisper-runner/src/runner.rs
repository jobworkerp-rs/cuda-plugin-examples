pub mod whisper;

use self::whisper::{model::WhisperLoaderImpl, segment::Segment, Task};
use crate::audio::trans::AudioTensorProcessor;
use anyhow::{anyhow, Result};
use candle_core::Device;
use candle_wrapper::{
    models::{token_id, ModelWrapper},
    whisper::WhisperWrapper,
};
use serde::Deserialize;
use symphonia::core::io::MediaSource;
use tokenizers::Tokenizer;
use whisper::Decoder;

#[derive(Debug, Deserialize, Clone)]
pub struct WhisperParams {
    /// The seed to use when generating random samples.
    pub seed: u64,

    /// Language.
    pub language: Option<String>,

    /// Task, when no task is specified, the input tokens contain only the sot token which can
    /// improve things when in no-timestamp mode.
    pub task: Task,

    /// Timestamps mode, this is not fully implemented yet.
    pub timestamps: bool,

    /// the track number for decode (for multiple audio track: default 0)
    pub n_tracks: usize,
}

pub struct WhisperRunModel {
    /// The model to embed sentences.(bert)
    model: WhisperWrapper,

    tokenizer: Tokenizer,

    device: Device,
    /// multilingual model or not
    pub is_multilingual: bool,
}

impl WhisperRunModel {
    pub fn new() -> Result<Self> {
        let model_loader = WhisperLoaderImpl::from_env()?;
        Self::new_from(&model_loader)
    }
    pub fn new_from(model_loader: &WhisperLoaderImpl) -> Result<Self> {
        let models = model_loader.load()?;
        let model = match models.model {
            ModelWrapper::Whisper(ww) => Ok(ww),
            m => Err(anyhow!("cannot apply this method to model: {:?}", m)),
        }?;
        let tokenizer = models.tokenizer;
        Ok(Self {
            model,
            tokenizer,
            device: models.device,
            is_multilingual: model_loader.model.is_multilingual(),
        })
    }

    pub fn decode_input(
        &mut self,
        data: Box<dyn MediaSource>,
        ext: Option<&String>,
        params: WhisperParams,
    ) -> Result<Vec<Segment>> {
        let buf_sec = whisper::CHUNK_LENGTH as u64;
        let n_mels = self.model.config().num_mel_bins;
        let processor =
            AudioTensorProcessor::new_by_data(data, ext, buf_sec, n_mels, params.n_tracks)?;
        self.decode(processor, params)
    }

    pub fn decode_file(&mut self, input_file: &str, params: WhisperParams) -> Result<Vec<Segment>> {
        let n_mels = self.model.config().num_mel_bins;
        let buf_sec = whisper::CHUNK_LENGTH as u64;
        let processor = AudioTensorProcessor::new_by_input(
            input_file.to_owned(),
            buf_sec,
            n_mels,
            params.n_tracks,
        )?;
        self.decode(processor, params)
    }
    pub fn decode(
        &mut self,
        mut processor: AudioTensorProcessor,
        params: WhisperParams,
    ) -> Result<Vec<Segment>> {
        let mel = processor.get_current_mel_tensor()?;
        let language_token = match (self.is_multilingual, params.language.as_ref()) {
            (true, None) => Some(self::whisper::multilingual::detect_language(
                &mut self.model,
                &self.tokenizer,
                &mel,
            )?),
            (false, None) => None,
            (true, Some(language)) => match token_id(&self.tokenizer, &format!("<|{language}|>")) {
                Ok(token_id) => Some(token_id),
                Err(_) => anyhow::bail!("language {language} is not supported"),
            },
            (false, Some(_)) => {
                anyhow::bail!("a language cannot be set for non-multilingual models")
            }
        };
        let mut dc = Decoder::new(
            self.model.clone(),
            self.tokenizer.clone(),
            params.seed,
            &self.device,
            language_token,
            params.task,
            params.timestamps,
        )?;
        let segs = dc.run(&mut processor)?;
        Ok(Self::merge_segs_with_ts(segs))
    }

    fn merge_segs_with_ts(segs: Vec<Segment>) -> Vec<Segment> {
        if let Some(last) = segs.last() {
            if !last.has_timestamp_tokens() {
                return segs;
            }
        }
        // use seg.divide_and_merge to divide and merge segments
        segs.into_iter().fold(Vec::<Segment>::new(), |acc, seg| {
            let mut acc = acc;
            if let Some(last) = acc.pop() {
                let mut ls = last.divide_and_merge(&seg);
                if !ls.is_empty() {
                    acc.append(&mut ls);
                }
                acc
            } else {
                acc.append(&mut seg.divide_by_ts_str());
                acc
            }
        })
    }
}
