use anyhow::{anyhow, Context, Result};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{self, BertModel, HiddenAct};
use candle_transformers::models::whisper;
use candle_wrapper::bert::BertWrapper;
use candle_wrapper::loader::HFModelLoader;
use candle_wrapper::models::{ConfigWrapper, HFModelSetting, ModelWrapper, Models};
use serde::Deserialize;
use serde::Serialize;
use std::path::{Path, PathBuf};

use crate::protobuf::embedding::EmbeddingOperation;

#[derive(PartialEq, Debug, Deserialize, Serialize, Clone)]
pub struct BertLoaderImpl {
    /// if quantized, cpu only (limited by candle)
    pub use_cpu: bool,

    pub model_id: String,

    /// L2 normalization for embeddings.
    pub normalize_embeddings: bool,

    /// Use tanh based approximation for Gelu instead of erf implementation.
    pub approximate_gelu: bool,

    /// use prefix for embedding key (e.g. "query: , passage: for multilingual-e5 model")
    pub prefix: Option<String>,
}
impl From<EmbeddingOperation> for BertLoaderImpl {
    fn from(op: EmbeddingOperation) -> Self {
        Self {
            use_cpu: op.use_cpu,
            model_id: op.model_id,
            normalize_embeddings: op.normalize_embeddings,
            approximate_gelu: op.approximate_gelu,
            prefix: op.prefix,
        }
    }
}

impl BertLoaderImpl {
    pub fn load(&self) -> Result<Models> {
        let setting: HFModelSetting = HFModelSetting::from_model_id(self.model_id.clone());
        self.load_from_setting(setting)
    }

    pub fn from_env() -> Result<BertLoaderImpl> {
        envy::prefixed("BERT_")
            .from_env::<BertLoaderImpl>()
            .context("cannot read bert model config from env:")
    }
}

impl HFModelLoader for BertLoaderImpl {
    fn load_config_inner(&self, config_file: &Path) -> Result<ConfigWrapper> {
        tracing::debug!("load model file: {}", &config_file.to_string_lossy());
        let config_file = std::fs::read(config_file.as_os_str())?;
        let model_config: bert::Config = serde_json::from_slice(&config_file)?;
        Ok(ConfigWrapper::Bert(model_config))
    }
    // TODO use proper load method by weight files extension
    fn load_model_inner(
        &self,
        config: &ConfigWrapper,
        weight_files: &[PathBuf],
    ) -> Result<(ModelWrapper, Device)> {
        match config {
            ConfigWrapper::Bert(conf) => {
                let device = Self::device(self.use_cpu)?;
                tracing::debug!("load model as Bert");
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(weight_files, whisper::DTYPE, &device)?
                };
                // let vb = if self.use_pth {
                //     VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
                // } else {
                //     unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
                // };
                let mut c = conf.clone();
                if self.approximate_gelu {
                    c.hidden_act = HiddenAct::GeluApproximate;
                }
                BertModel::load(vb, &c)
                    .map(|m| (ModelWrapper::Bert(BertWrapper::Bert(m)), device))
                    .map_err(|e| anyhow!("{:?}", e))
            }
            m => {
                tracing::error!("cannot load whisper model as bert: {:?}", m);
                Err(anyhow!("cannot load whisper model as bert: {:?}", m))
            }
        }
    }
}
