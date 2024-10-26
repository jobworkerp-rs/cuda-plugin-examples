pub mod llm;

use std::io::Cursor;

use anyhow::{anyhow, Context, Result};
use candle_wrapper::PluginRunner;
use command_utils::util::result::FlatMap;
use llm::model::LLMModelLoaderConfig;
use llm::LlmInferencer;
use prost::Message;
use protobuf::llm::{CandleLlmArg, CandleLlmOperation};

pub mod protobuf {
    pub mod llm {
        include!(concat!(env!("OUT_DIR"), "/llm.rs"));
    }
}

// suppress warn improper_ctypes_definitions
#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn load_plugin() -> Box<dyn PluginRunner + Send + Sync> {
    dotenvy::dotenv().ok();
    let plugin = LlmRunnerPlugin::new();
    Box::new(plugin)
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn free_plugin(ptr: Box<dyn PluginRunner + Send + Sync>) {
    drop(ptr);
}

pub struct LlmRunnerPlugin {
    inferencer: Option<LlmInferencer>,
}

impl LlmRunnerPlugin {
    pub fn new() -> Self {
        Self { inferencer: None }
    }
    pub fn load_config_from_env(&mut self) -> Result<()> {
        let conf = dotenvy::var("HF_MODEL_LOADER_PROFILE")
            .context("profile is not specified. read other config")
            .flat_map(|s| LLMModelLoaderConfig::find_hf_model_loader_from_profile_name(&s))
            .or_else(|_| envy::prefixed("LLM_").from_env::<LLMModelLoaderConfig>())
            .context("cannot read model config from env:")?;
        self.inferencer = Some(LlmInferencer::new(conf)?);
        Ok(())
    }
    pub fn load_config(operation: CandleLlmOperation) -> Result<LLMModelLoaderConfig> {
        if let Some(profile) = operation.loader_profile.as_ref() {
            LLMModelLoaderConfig::find_hf_model_loader_from_profile_name(profile)
        } else {
            let config = LLMModelLoaderConfig {
                profile: None,
                model_id: operation.model_id,
                revision: operation.revision,
                tokenizer_file: operation.tokenizer_file,
                weight_files: if operation.weight_files.is_empty() {
                    None
                } else {
                    Some(operation.weight_files)
                },
                model_config_file: operation.model_config_file,
                quantized: Some(operation.quantized),
                use_flash_attn: operation.use_flash_attn,
                eos_token: operation.eos_token.unwrap_or("<s>".to_string()),
                use_cpu: operation.use_cpu,
            };
            Ok(config)
        }
    }
}

impl PluginRunner for LlmRunnerPlugin {
    fn name(&self) -> String {
        // specify as same string as worker.operation
        String::from("llm.CandleLlm")
    }
    fn load(&mut self, operation: Vec<u8>) -> Result<()> {
        let op = CandleLlmOperation::decode(&mut Cursor::new(operation))
            .map_err(|e| anyhow!("decode error: {}", e))?;
        let config = Self::load_config(op)?;
        self.inferencer = Some(LlmInferencer::new(config)?);
        Ok(())
    }
    fn operation_proto(&self) -> String {
        include_str!("../protobuf/llm/llm_operation.proto").to_string()
    }
    fn job_args_proto(&self) -> String {
        include_str!("../protobuf/llm/llm_arg.proto").to_string()
    }
    // return string
    fn result_output_proto(&self) -> Option<String> {
        None
    }
    // if true, use job result of before job, else use job args from request
    fn use_job_result(&self) -> bool {
        false
    }
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>> {
        if let Some(inferencer) = self.inferencer.as_mut() {
            let args = CandleLlmArg::decode(&mut Cursor::new(arg))
                .map_err(|e| anyhow!("decode error: {}", e))?;
            let text = inferencer
                .generate_text(args.into())
                .context("failed to decode")?;
            tracing::debug!("END OF LLMRunner: {text:?}",);
            // serialize and return
            Ok(vec![text.bytes().collect()])
        } else {
            Err(anyhow!("inferencer is not loaded"))
        }
    }
    fn cancel(&self) -> bool {
        tracing::warn!("LLMRunner cancel: not implemented!");
        false
    }
}
