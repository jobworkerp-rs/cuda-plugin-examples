pub mod llm;

use std::io::Cursor;

use anyhow::{anyhow, Context, Result};
use command_utils::util::result::FlatMap;
use llm::model::LLMModelLoaderConfig;
use llm::LlmInferencer;
use prost::Message;

pub mod protobuf {
    pub mod llm {
        include!(concat!(env!("OUT_DIR"), "/llm.rs"));
    }
}

pub trait PluginRunner: Send + Sync {
    fn name(&self) -> String;
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>>;
    fn cancel(&self) -> bool;
}

// suppress warn improper_ctypes_definitions
#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn load_plugin() -> Box<dyn PluginRunner + Send + Sync> {
    dotenvy::dotenv().ok();
    let plugin = LlmRunnerPlugin::new().expect("failed to load plugin");
    Box::new(plugin)
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn free_plugin(ptr: Box<dyn PluginRunner + Send + Sync>) {
    drop(ptr);
}

pub struct LlmRunnerPlugin {
    inferencer: LlmInferencer,
}
// static DATA: OnceCell<Bytes> = OnceCell::new();

impl LlmRunnerPlugin {
    pub fn new() -> Result<Self> {
        let config = Self::load_config_from_env()?;
        Ok(Self {
            inferencer: LlmInferencer::new(config.clone())?,
        })
    }
    fn load_config_from_env() -> Result<LLMModelLoaderConfig> {
        dotenvy::var("HF_MODEL_LOADER_PROFILE")
            .context("profile is not specified. read other config")
            .flat_map(|s| LLMModelLoaderConfig::find_hf_model_loader_from_profile_name(&s))
            .or_else(|_| envy::prefixed("LLM_").from_env::<LLMModelLoaderConfig>())
            .context("cannot read model config from env:")
    }
}

impl PluginRunner for LlmRunnerPlugin {
    fn name(&self) -> String {
        // specify as same string as worker.operation
        String::from("LLMPromptRunner")
    }
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>> {
        let args = protobuf::llm::InferenceRequest::decode(&mut Cursor::new(arg))
            .map_err(|e| anyhow!("decode error: {}", e))?;
        let text = self
            .inferencer
            .generate_text(args.into())
            .context("failed to decode")?;
        tracing::debug!("END OF LLMRunner: {text:?}",);
        // serialize and return
        let res = protobuf::llm::InferenceResponse { text };
        let mut buf = Vec::with_capacity(res.encoded_len());
        res.encode(&mut buf).unwrap();
        Ok(vec![buf])
    }
    fn cancel(&self) -> bool {
        tracing::warn!("LLMRunner cancel: not implemented!");
        false
    }
}
