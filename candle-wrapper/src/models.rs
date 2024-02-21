use crate::bert::BertWrapper;
use crate::llm::config::LLMConfigWrapper;
use crate::llm::LLMModelWrapper;
use crate::whisper::WhisperWrapper;
use anyhow::{anyhow, Result};
use candle_core::Device;
use candle_transformers::models::bert;
use candle_transformers::models::whisper;
use serde::Deserialize;
use serde::Serialize;
use tokenizers::Tokenizer;

// typical weight and setting filenames
pub const WEIGHT_FILES_SAFETENSOR_2FILES: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];
pub const WEIGHT_FILES_SAFETENSOR_SINGLE: [&str; 1] = ["model.safetensors"];
pub const WEIGHT_FILES_GGUF_SINGLE: [&str; 1] = ["model-q4k.gguf"]; // not used

pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const MODEL_CONFIG_FILE: &str = "config.json";

pub const MAIN_REVISION: &str = "main";

pub const DEFAULT_QLEVEL: &str = "q80";

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    match tokenizer.token_to_id(token) {
        None => Err(anyhow!("no token-id for {token}")),
        Some(id) => Ok(id),
    }
}

#[derive(Debug, Clone)]
pub enum ConfigWrapper {
    Bert(bert::Config),
    Whisper(whisper::Config),
    LLM(LLMConfigWrapper),
}

impl From<ConfigWrapper> for bert::Config {
    fn from(c: ConfigWrapper) -> Self {
        match c {
            ConfigWrapper::Bert(c) => c,
            _ => panic!("not bert config:{:?}", c),
        }
    }
}
impl From<ConfigWrapper> for whisper::Config {
    fn from(c: ConfigWrapper) -> Self {
        match c {
            ConfigWrapper::Whisper(c) => c,
            _ => panic!("not whisper config:{:?}", c),
        }
    }
}
impl From<ConfigWrapper> for LLMConfigWrapper {
    fn from(c: ConfigWrapper) -> Self {
        match c {
            ConfigWrapper::LLM(c) => c,
            _ => panic!("not llm config:{:?}", c),
        }
    }
}

#[derive(Debug)]
pub enum ModelWrapper {
    Bert(BertWrapper),
    Whisper(WhisperWrapper),
    LLM(LLMModelWrapper),
}

// TODO change file types (pth, safetensor, onnx): only safetensor
// const definitions for each model are only availables
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HFModelSetting {
    pub model_id: Option<String>,       // if None, use local file
    pub revision: Option<String>,       // default: main
    pub tokenizer_file: Option<String>, // default: tokenizer.json
    pub weight_files: Option<Vec<String>>,
    pub model_config_file: Option<String>, // default: config.json
}

impl HFModelSetting {
    pub fn new(
        model_id: Option<String>,
        revision: Option<String>,
        tokenizer_file: Option<String>,
        weight_files: Option<Vec<String>>,
        model_config_file: Option<String>,
    ) -> Self {
        Self {
            model_id,
            revision,
            tokenizer_file,
            weight_files,
            model_config_file,
        }
    }
    // normal model setting with huggingface model_id (predefined model_id only)
    pub fn from_model_id(model_id: String) -> Self {
        Self {
            model_id: Some(model_id),
            revision: None,
            tokenizer_file: None,
            weight_files: None,
            model_config_file: None,
        }
    }
    // use huggingface model_id with quantized setting: model_size_postfix, qlevel
    // (single weight file only)
    pub fn from(
        model_id: String,
        revision: Option<String>,
        quantized: bool,
        model_size_postfix: Option<&String>,
        qlevel: Option<&String>,
    ) -> Self {
        let (config, tokenizer, model) =
            Self::model_filenames(quantized, model_size_postfix, qlevel);
        Self {
            model_id: Some(model_id),
            revision,
            tokenizer_file: Some(tokenizer),
            weight_files: Some(vec![model]),
            model_config_file: Some(config),
        }
    }
    // typical (config, tokenizer, model) filenames
    pub fn model_filenames(
        quantized: bool,
        model_size_postfix: Option<&String>,
        qlevel: Option<&String>,
    ) -> (String, String, String) {
        let model_ext = if quantized { "gguf" } else { "safetensors" }; // TODO support onyx ?
        let sz = if let Some(ext) = model_size_postfix {
            format!("-{}", ext)
        } else {
            "".to_string()
        };
        // only add postfix for quantized model
        let pf = if quantized {
            let p = qlevel.cloned().unwrap_or(DEFAULT_QLEVEL.to_string());
            format!("-{p}")
        } else {
            "".to_string()
        };
        (
            format!("config{sz}.json"),
            format!("tokenizer{sz}.json"),
            format!("model{sz}{pf}.{model_ext}"),
        )
    }
}

pub struct Models {
    pub model: ModelWrapper,
    pub tokenizer: Tokenizer,
    pub device: Device,
}
