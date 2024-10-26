use super::tokenizer::TokenOutputStream;
use anyhow::{anyhow, Context, Result};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::DType;
use candle_core::Device;
use candle_transformers::models::falcon;
use candle_transformers::models::llama;
use candle_transformers::models::mistral;
use candle_transformers::models::stable_lm;
use candle_wrapper::llm::config::LLMConfigWrapper;
use candle_wrapper::llm::config::EOS_TOKEN;
use candle_wrapper::llm::config::{LLMConfig, ModelType};
use candle_wrapper::llm::LLMModelWrapper;
use candle_wrapper::loader::HFModelLoader;
use candle_wrapper::models::ConfigWrapper;
use candle_wrapper::models::HFModelSetting;
use candle_wrapper::models::ModelWrapper;
use command_utils::util::option::Exists;
use command_utils::util::result::FlatMap;
use once_cell::sync::Lazy;
use serde::Deserialize;
use serde::Serialize;
use std::path::{Path, PathBuf};
// use candle_transformers::models::quantized_stable_lm;

// const definitions for each model are only availables
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LLMModelLoaderConfig {
    pub profile: Option<ModelLoaderProfile>,
    pub model_id: Option<String>,
    pub revision: Option<String>,       // default: main
    pub tokenizer_file: Option<String>, // default: tokenizer.json
    pub weight_files: Option<Vec<String>>,
    pub model_config_file: Option<String>, // default: config.json
    pub quantized: Option<bool>,           // TODO not implemented
    pub use_flash_attn: bool,              // available for llama or mistral
    pub eos_token: String,                 // predefined (TODO load from tokenizer.json)
    pub use_cpu: bool,                     // if quantized, cpu only (limited by candle)
}

#[derive(Debug, Clone)]
pub struct LLMModelLoader {
    pub config: LLMModelLoaderConfig,
}

impl Default for LLMModelLoaderConfig {
    fn default() -> Self {
        STABLE_LM_JA_3B_4E1T.clone()
    }
}

// typical weight filenames
static WEIGHT_FILES_SAFETENSOR_7B: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];
const WEIGHT_FILES_SAFETENSOR_3B: [&str; 1] = ["model.safetensors"];
//const WEIGHT_FILES_GGUF_3B: [&'static str; 1] = ["model-q4k.gguf"]; // not used
//const TOKENIZER_FILE: &'static str = "tokenizer.json";
const MODEL_FILE: &str = "config.json";
const MAIN_REVISION: &str = "main";

// // typical llm model definitions
// need to log-in huggingface account which agree to meta license
static LLAMA_7B_V2: Lazy<LLMModelLoaderConfig> = Lazy::new(|| LLMModelLoaderConfig {
    profile: Some(ModelLoaderProfile::Llama7bV2),
    model_id: Some("meta-llama/Llama-2-7b-hf".to_string()),
    revision: Some(MAIN_REVISION.to_string()),
    tokenizer_file: None, // use default from hf cache
    weight_files: Some(WEIGHT_FILES_SAFETENSOR_7B.map(|s| s.to_string()).to_vec()),
    model_config_file: Some(MODEL_FILE.to_string()),
    quantized: Some(false),
    use_flash_attn: true,
    eos_token: "<s>".to_string(),
    use_cpu: false,
});
// XXX need to convert pytorch bin to safetensor manually
static ELYZA_7B_FAST: Lazy<LLMModelLoaderConfig> = Lazy::new(|| LLMModelLoaderConfig {
    profile: Some(ModelLoaderProfile::Elyza7bFast),
    model_id: Some("elyza/ELYZA-japanese-Llama-2-7b-fast".to_string()),
    revision: Some(MAIN_REVISION.to_string()),
    tokenizer_file: None, // use default from hf cache
    weight_files: Some(WEIGHT_FILES_SAFETENSOR_7B.map(|s| s.to_string()).to_vec()),
    model_config_file: Some(MODEL_FILE.to_string()),
    quantized: Some(false),
    use_flash_attn: true,
    eos_token: EOS_TOKEN.to_string(),
    use_cpu: false,
});
static STABLE_LM_JA_VOCAB_BETA_7B: Lazy<LLMModelLoaderConfig> =
    Lazy::new(|| LLMModelLoaderConfig {
        profile: Some(ModelLoaderProfile::StableLmJaVocabBeta7b),
        model_id: Some("stabilityai/japanese-stablelm-base-ja_vocab-beta-7b".to_string()),
        revision: Some(MAIN_REVISION.to_string()),
        tokenizer_file: None, // use default from hf cache
        weight_files: Some(WEIGHT_FILES_SAFETENSOR_7B.map(|s| s.to_string()).to_vec()),
        model_config_file: Some(MODEL_FILE.to_string()),
        quantized: Some(false),
        use_flash_attn: true,
        eos_token: EOS_TOKEN.to_string(),
        use_cpu: false,
    });
static STABLE_LM_JA_GAMMA_7B: Lazy<LLMModelLoaderConfig> = Lazy::new(|| LLMModelLoaderConfig {
    profile: Some(ModelLoaderProfile::StableLmJaGamma7b),
    model_id: Some("stabilityai/japanese-stablelm-base-gamma-7b".to_string()),
    revision: Some(MAIN_REVISION.to_string()),
    tokenizer_file: None, // use default from hf cache
    weight_files: Some(WEIGHT_FILES_SAFETENSOR_7B.map(|s| s.to_string()).to_vec()),
    model_config_file: Some(MODEL_FILE.to_string()),
    quantized: Some(false),
    use_flash_attn: true,
    eos_token: EOS_TOKEN.to_string(),
    use_cpu: false,
});
static STABLE_LM_JA_3B_4E1T: Lazy<LLMModelLoaderConfig> = Lazy::new(|| LLMModelLoaderConfig {
    profile: Some(ModelLoaderProfile::StableLmJa3b4e1t),
    model_id: Some("stabilityai/japanese-stablelm-3b-4e1t-base".to_string()),
    revision: Some(MAIN_REVISION.to_string()),
    tokenizer_file: None, // use default from hf cache
    weight_files: Some(WEIGHT_FILES_SAFETENSOR_3B.map(|s| s.to_string()).to_vec()),
    model_config_file: Some(MODEL_FILE.to_string()),
    quantized: Some(false),
    use_flash_attn: true,
    eos_token: "<|endoftext|>".to_string(),
    use_cpu: false,
});

// predefined model loader profile (Just for me :))
#[derive(PartialEq, Debug, Deserialize, Serialize, Clone)]
pub enum ModelLoaderProfile {
    Llama7bV2,
    Elyza7bFast,
    StableLmJaVocabBeta7b,
    StableLmJaGamma7b,
    StableLmJa3b4e1t,
}
impl ModelLoaderProfile {
    pub fn from_name(name: &str) -> Option<ModelLoaderProfile> {
        match name {
            "llama-7b-v2" => Some(ModelLoaderProfile::Llama7bV2),
            "elyza-7b-fast" => Some(ModelLoaderProfile::Elyza7bFast),
            "stable-lm-ja-vocab-beta-7b" => Some(ModelLoaderProfile::StableLmJaVocabBeta7b),
            "stable-lm-ja-gamma-7b" => Some(ModelLoaderProfile::StableLmJaGamma7b),
            "stable-lm-ja-3b-4e1t" => Some(ModelLoaderProfile::StableLmJa3b4e1t),
            _ => None,
        }
    }
}

static ALL_MODEL_LOADER_PROFILES: Lazy<Vec<LLMModelLoaderConfig>> = Lazy::new(|| {
    vec![
        LLAMA_7B_V2.clone(),
        ELYZA_7B_FAST.clone(),
        STABLE_LM_JA_VOCAB_BETA_7B.clone(),
        STABLE_LM_JA_GAMMA_7B.clone(),
        STABLE_LM_JA_3B_4E1T.clone(),
    ]
});

pub struct LLM {
    pub model: LLMModelWrapper,
    pub tokenizer: TokenOutputStream,
    pub device: Device,
}

impl LLMModelLoaderConfig {
    pub fn find_hf_model_loader_from_profile_name(s: &String) -> Result<LLMModelLoaderConfig> {
        ModelLoaderProfile::from_name(s.as_str())
            .ok_or(anyhow!("profile not found:{}", &s))
            .context("serde_json::to_string")
            .flat_map(|p| {
                LLMModelLoaderConfig::find_hf_model_loader(&p).ok_or(anyhow!("not found"))
            })
    }
    pub fn find_hf_model_loader(profile: &ModelLoaderProfile) -> Option<LLMModelLoaderConfig> {
        ALL_MODEL_LOADER_PROFILES
            .iter()
            .find(|m| m.profile.as_ref().exists(|p| p == profile))
            .cloned()
    }

    pub fn model_setting(&self) -> HFModelSetting {
        HFModelSetting {
            model_id: self.model_id.clone(),
            revision: self.revision.clone(),
            tokenizer_file: self.tokenizer_file.clone(),
            weight_files: self.weight_files.clone(),
            model_config_file: self.model_config_file.clone(),
        }
    }
}
impl LLMModelLoader {
    pub fn new(config: LLMModelLoaderConfig) -> Self {
        Self { config }
    }
    // GPU or CPU (GPU is only available for cuda id:0 device)
    pub fn device(cpu: bool) -> Result<Device> {
        if cpu {
            Ok(Device::Cpu)
        } else if cuda_is_available() {
            Ok(Device::new_cuda(0)?)
        } else if metal_is_available() {
            Ok(Device::new_metal(0)?)
        } else {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                tracing::info!(
                    "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
                );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                tracing::info!(
                    "Running on CPU, to run on GPU, build this example with `--features cuda`"
                );
            }
            Ok(Device::Cpu)
        }
    }

    // TODO resolve from model_config
    fn dtype(&self, device: &Device) -> DType {
        if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        }
    }

    // devide each model to each files
    pub fn load(&self) -> Result<LLM> {
        let models = self.load_from_setting(self.config.model_setting())?;
        tracing::debug!("llm model loaded: {:?}", &self.config.model_id);
        let model: LLMModelWrapper = match models.model {
            ModelWrapper::LLM(m) => m,
            m => anyhow::bail!("cannot load model as llm: {:?}", m),
        };
        Ok(LLM {
            model,
            tokenizer: TokenOutputStream::new(models.tokenizer),
            device: models.device,
        })
    }
}

impl HFModelLoader for LLMModelLoader {
    // implement HFModelLoader load_config_inner and load_model_inner trait for LLMModelLoader
    fn load_config_inner(&self, config_file: &Path) -> Result<ConfigWrapper> {
        let mut config: LLMConfig =
            serde_json::from_slice(&std::fs::read(config_file.as_os_str())?)
                .context("load model config")?;
        // stable_lm: cannot init (field pub(crate)) but can deserialize
        if config.model_type() == Some(ModelType::StableLM) {
            // re-read as stable_lm::Config for deserialize only
            let mut c: stable_lm::Config =
                serde_json::from_slice(&std::fs::read(config_file.as_os_str())?)
                    .context("load stable_lm model config")?;
            c.set_use_flash_attn(self.config.use_flash_attn);
            Ok(ConfigWrapper::LLM(c.into()))
        } else {
            config.use_flash_attn = Some(self.config.use_flash_attn);
            Ok(ConfigWrapper::LLM(config.into()))
        }
    }
    // TODO use proper load method by weight files extension (quontized)
    fn load_model_inner(
        &self,
        config: &ConfigWrapper,
        weight_files: &[PathBuf],
    ) -> Result<(candle_wrapper::models::ModelWrapper, Device)> {
        let config: LLMConfigWrapper = config.clone().into();
        let device = Self::device(self.config.use_cpu)?;
        let model = match config {
            LLMConfigWrapper::Falcon(config) => {
                tracing::debug!("load model as Falcon");
                let vb = unsafe {
                    candle_nn::VarBuilder::from_mmaped_safetensors(
                        weight_files,
                        self.dtype(&device),
                        &device,
                    )?
                };
                config.validate()?;
                let model = falcon::Falcon::load(vb, config)?;
                LLMModelWrapper::Falcon(model)
            }
            LLMConfigWrapper::Llama(config) => {
                tracing::debug!("load model as LLaMa");
                // XXX use_kv_cache: always true
                let cache = llama::Cache::new(true, self.dtype(&device), &config, &device)?;
                let vb = unsafe {
                    candle_nn::VarBuilder::from_mmaped_safetensors(
                        weight_files,
                        self.dtype(&device),
                        &device,
                    )?
                };
                let model = llama::Llama::load(vb, &config)?;
                LLMModelWrapper::Llama(model, cache)
            }
            LLMConfigWrapper::Mistral(config) => {
                tracing::debug!("load model as Mistral");
                let vb = unsafe {
                    candle_nn::VarBuilder::from_mmaped_safetensors(
                        weight_files,
                        self.dtype(&device),
                        &device,
                    )?
                };
                let model = mistral::Model::new(&config, vb)?;
                LLMModelWrapper::Mistral(model)
            }
            LLMConfigWrapper::StableLM(config) => {
                tracing::debug!("load model as StableLM");
                let vb = unsafe {
                    candle_nn::VarBuilder::from_mmaped_safetensors(
                        weight_files,
                        self.dtype(&device),
                        &device,
                    )?
                };
                let model = stable_lm::Model::new(&config, vb)?;
                LLMModelWrapper::StableLM(model)
            } // TODO Quantum model is not supported yet
              // LLMConfigWrapper::QStableLM(config) => {
              //     //FIXME not working
              //     let device = Self::device(self.use_cpu)?;
              //     let filename = weight_files
              //         .first()
              //         .ok_or::<anyhow::Error>(LLMErrors::ModelNotFound.into())?;
              //     let vb =
              //         candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename)?;
              //     let model = quantized_stable_lm::Model::new(&config, vb)?;
              //     Ok((ModelWrapper::QStableLM(model), device))
              // }
        };
        Ok((ModelWrapper::LLM(model), device))
    }
}
