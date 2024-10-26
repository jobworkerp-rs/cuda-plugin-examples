// https://github.com/openai/whisper/blob/main/whisper/model.py/rgs
// TODO:
// - Batch size greater than 1.
// - More token filters (SuppressBlanks, ApplyTimestampRules).

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

use anyhow::{anyhow, Context, Result};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m};
use candle_wrapper::loader::HFModelLoader;
use candle_wrapper::models::{ConfigWrapper, HFModelSetting, ModelWrapper, Models};
use candle_wrapper::whisper::WhisperWrapper;
use clap::ValueEnum;
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::protobuf::whisper::WhisperOperation;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, ValueEnum)]
pub enum WhichModel {
    Tiny,
    #[value(name = "tiny.en")]
    TinyEn,
    Base,
    #[value(name = "base.en")]
    BaseEn,
    Small,
    #[value(name = "small.en")]
    SmallEn,
    Medium,
    #[value(name = "medium.en")]
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    #[value(name = "distil-medium.en")]
    DistilMediumEn,
    #[value(name = "distil-large-v2")]
    DistilLargeV2,
}
impl From<String> for WhichModel {
    fn from(value: String) -> Self {
        match value.as_str() {
            "tiny" => Self::Tiny,
            "tiny.en" => Self::TinyEn,
            "base" => Self::Base,
            "base.en" => Self::BaseEn,
            "small" => Self::Small,
            "small.en" => Self::SmallEn,
            "medium" => Self::Medium,
            "medium.en" => Self::MediumEn,
            "large" => Self::Large,
            "large-v2" => Self::LargeV2,
            "large-v3" => Self::LargeV3,
            "distil-medium.en" => Self::DistilMediumEn,
            "distil-large-v2" => Self::DistilLargeV2,
            _ => Self::LargeV3, // default
        }
    }
}

impl WhichModel {
    pub fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3 => true,
            Self::TinyEn
            | Self::BaseEn
            | Self::SmallEn
            | Self::MediumEn
            | Self::DistilMediumEn
            | Self::DistilLargeV2 => false,
        }
    }

    pub fn model_and_revision(&self, quantized: bool) -> Option<(&'static str, &'static str)> {
        if quantized {
            match self {
                Self::TinyEn | Self::Tiny => Some(("lmz/candle-whisper", "main")),
                _ => None,
            }
        } else {
            match self {
                Self::Tiny => Some(("openai/whisper-tiny", "main")),
                Self::TinyEn => Some(("openai/whisper-tiny.en", "refs/pr/15")),
                Self::Base => Some(("openai/whisper-base", "refs/pr/22")),
                Self::BaseEn => Some(("openai/whisper-base.en", "refs/pr/13")),
                Self::Small => Some(("openai/whisper-small", "main")),
                Self::SmallEn => Some(("openai/whisper-small.en", "refs/pr/10")),
                Self::Medium => Some(("openai/whisper-medium", "main")),
                Self::MediumEn => Some(("openai/whisper-medium.en", "main")),
                Self::Large => Some(("openai/whisper-large", "refs/pr/36")),
                Self::LargeV2 => Some(("openai/whisper-large-v2", "refs/pr/57")),
                Self::LargeV3 => Some(("openai/whisper-large-v3", "main")),
                Self::DistilMediumEn => Some(("distil-whisper/distil-medium.en", "main")),
                Self::DistilLargeV2 => Some(("distil-whisper/distil-large-v2", "main")),
            }
        }
    }
}

#[derive(PartialEq, Debug, Deserialize, Clone)]
pub struct WhisperLoaderImpl {
    /// if quantized, cpu only (limited by candle)
    pub use_cpu: bool,

    pub quantized: bool,

    pub model: WhichModel,
}
impl From<WhisperOperation> for WhisperLoaderImpl {
    fn from(op: WhisperOperation) -> Self {
        Self {
            use_cpu: op.use_cpu,
            quantized: op.quantized,
            model: op.model.map(|v| v.into()).unwrap_or(WhichModel::LargeV3),
        }
    }
}

impl WhisperLoaderImpl {
    pub fn load(&self) -> Result<Models> {
        let ms_pfix = if self.quantized {
            // XXX for model "lmz/candle-whisper"
            match self.model {
                WhichModel::TinyEn => Ok(Some("tiny-en".to_string())),
                WhichModel::Tiny => Ok(Some("tiny".to_string())),
                _ => Err(anyhow!("no quantized support for {:?}", &self.model)),
            }?
        } else {
            None
        };
        let qlevel = if self.quantized {
            Some("q80".to_string()) // XXX candle default
        } else {
            None
        };
        let (model_id, revision) = self
            .model
            .model_and_revision(self.quantized)
            .ok_or(anyhow!("no model for {:?}", &self.model))?;

        let setting: HFModelSetting = HFModelSetting::from(
            model_id.to_string(),
            Some(revision.to_string()),
            self.quantized,
            ms_pfix.as_ref(),
            qlevel.as_ref(),
        );
        self.load_from_setting(setting)
    }

    pub fn from_env() -> Result<WhisperLoaderImpl> {
        envy::prefixed("WHISPER_")
            .from_env::<WhisperLoaderImpl>()
            .context("cannot read whisper model config from env:")
    }
}

impl HFModelLoader for WhisperLoaderImpl {
    fn load_config_inner(&self, config_file: &Path) -> Result<ConfigWrapper> {
        tracing::debug!("load model file: {}", &config_file.to_string_lossy());
        let config_file = std::fs::read(config_file.as_os_str())?;
        let model_config: m::Config = serde_json::from_slice(&config_file)?;
        Ok(ConfigWrapper::Whisper(model_config))
    }
    // TODO use proper load method by weight files extension
    fn load_model_inner(
        &self,
        config: &ConfigWrapper,
        weight_files: &[PathBuf],
    ) -> Result<(ModelWrapper, Device)> {
        match config {
            ConfigWrapper::Whisper(conf) => {
                let device = Self::device(self.use_cpu)?;
                tracing::debug!("load model as Whisper");
                let model = if self.quantized {
                    let wf = weight_files.first().ok_or(anyhow!("no weight file"))?;
                    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                        wf, &device,
                    )?;
                    WhisperWrapper::QWhisper(m::quantized_model::Whisper::load(&vb, conf.clone())?)
                } else {
                    let vb = unsafe {
                        VarBuilder::from_mmaped_safetensors(weight_files, m::DTYPE, &device)?
                    };
                    WhisperWrapper::Whisper(m::model::Whisper::load(&vb, conf.clone())?)
                };
                Ok((ModelWrapper::Whisper(model), device))
            }
            _ => Err(anyhow!("cannot load model as Whisper. config={:?}", config)),
        }
    }
}
