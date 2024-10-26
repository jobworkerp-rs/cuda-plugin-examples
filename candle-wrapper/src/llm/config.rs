use candle_nn::Activation;
use candle_transformers::models::falcon;
use candle_transformers::models::llama;
use candle_transformers::models::mistral;
use candle_transformers::models::stable_lm;
use serde::Deserialize;
use serde::Serialize;

// TODO from config
pub const EOS_TOKEN: &str = "</s>";

// TODO divide quantized
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub enum ModelType {
    Falcon,
    Llama,
    Mistral,
    StableLM,
}

// use for deserializing config.json
#[derive(Deserialize, Clone, Debug)]
pub struct LLMConfig {
    pub alibi: Option<bool>,                    // falcon
    pub architectures: Vec<String>,             // not used now
    pub attention_dropout: Option<f64>,         // falcon
    pub bias: Option<bool>,                     // falcon
    pub bos_token_id: u32,                      // not used now
    pub eos_token_id: u32,                      // not used now
    pub hidden_act: Option<String>,             // stable_lm, mistral  (exists other model but used)
    pub hidden_dropout: Option<f64>,            // falcon
    pub hidden_size: usize,                     // llama, falcon, stable_lm, mistral
    pub initializer_range: Option<f64>,         // falcon
    pub intermediate_size: Option<usize>,       // llama, stable_lm, mistral
    pub layer_norm_epsilon: Option<f64>,        // falcon
    pub n_head_kv: Option<usize>,               // falcon (None(fixed))
    pub max_position_embeddings: Option<usize>, // stable_lm, mistral
    pub multi_query: Option<bool>,              // falcon
    pub new_decoder_architecture: Option<bool>, // falcon
    pub norm_eps: Option<f64>,                  // stable_lm
    pub num_attention_heads: usize,             // llama, falcon, stable_lm, mistral
    pub num_hidden_layers: usize,               // llama, falcon, stable_lm, mistral
    pub num_key_value_heads: Option<usize>,     // llama, stable_lm, mistral
    pub parallel_attn: Option<bool>,            // falcon
    pub rms_norm_eps: Option<f64>,              // llama, mistral
    pub rope_pct: Option<f64>,                  // stable_lm
    #[serde(default = "default_rope")]
    pub rope_theta: f32, // llama, stable_lm, mistral
    pub use_cache: Option<bool>,                // falcon, stable_lm
    pub vocab_size: usize,                      // llama, falcon, stable_lm, mistral

    pub sliding_window: Option<usize>, // mistral

    pub use_flash_attn: Option<bool>, // for candle (not include in config.json)
}

fn default_rope() -> f32 {
    10_000.0
}

impl LLMConfig {
    pub fn model_type(&self) -> Option<ModelType> {
        // map from architectures
        match self.architectures[0].as_str() {
            "FalconForCausalLM" => Some(ModelType::Falcon),
            "LlamaForCausalLM" => Some(ModelType::Llama),
            "MistralForCausalLM" => Some(ModelType::Mistral),
            "StableLMEpochForCausalLM" => Some(ModelType::StableLM),
            _ => {
                tracing::error!("unknown model type: {:?}", &self.architectures);
                None
            }
        }
    }
}

impl From<LLMConfig> for falcon::Config {
    fn from(config: LLMConfig) -> falcon::Config {
        let default = falcon::Config::default();
        tracing::debug!("load falcon model params: {:?}", &config);
        falcon::Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            layer_norm_epsilon: config
                .layer_norm_epsilon
                .unwrap_or(default.layer_norm_epsilon),
            initializer_range: config
                .initializer_range
                .unwrap_or(default.initializer_range),
            use_cache: config.use_cache.unwrap_or(default.use_cache),
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
            hidden_dropout: config.hidden_dropout.unwrap_or(default.hidden_dropout),
            attention_dropout: config
                .attention_dropout
                .unwrap_or(default.attention_dropout),
            n_head_kv: config.n_head_kv,
            alibi: config.alibi.unwrap_or(default.alibi),
            new_decoder_architecture: config
                .new_decoder_architecture
                .unwrap_or(default.new_decoder_architecture),
            multi_query: config.multi_query.unwrap_or(default.multi_query),
            parallel_attn: config.parallel_attn.unwrap_or(default.parallel_attn),
            bias: config.bias.unwrap_or(default.bias),
        }
    }
}

impl From<LLMConfig> for llama::Config {
    fn from(config: LLMConfig) -> llama::Config {
        tracing::debug!("load llama model params: {:?}", &config);
        // default not exists
        let default = llama::Config::config_7b_v2(true);
        llama::Config {
            hidden_size: config.hidden_size,
            intermediate_size: config
                .intermediate_size
                .unwrap_or(default.intermediate_size),
            vocab_size: config.vocab_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config
                .num_key_value_heads
                .unwrap_or(default.num_key_value_heads),
            rms_norm_eps: config.rms_norm_eps.unwrap_or(default.rms_norm_eps),
            rope_theta: config.rope_theta,
            use_flash_attn: config.use_flash_attn.unwrap_or(true),
            bos_token_id: Some(config.bos_token_id),
            eos_token_id: Some(llama::LlamaEosToks::Single(config.eos_token_id)), // TODO multi
            rope_scaling: None,                                                   // TODO
            max_position_embeddings: config
                .max_position_embeddings
                .unwrap_or(llama::DEFAULT_MAX_SEQ_LEN),
            tie_word_embeddings: false, // TODO
        }
    }
}

// TODO
impl From<LLMConfig> for mistral::Config {
    fn from(value: LLMConfig) -> Self {
        let default = mistral::Config::config_7b_v0_1(true);
        // XXX cannot init mistral::Config (field scope "crate")
        mistral::Config {
            vocab_size: value.vocab_size,
            hidden_size: value.hidden_size,
            intermediate_size: value.intermediate_size.unwrap_or(default.intermediate_size),
            num_hidden_layers: value.num_hidden_layers,
            num_attention_heads: value.num_attention_heads,
            num_key_value_heads: value
                .num_key_value_heads
                .unwrap_or(default.num_key_value_heads),
            hidden_act: value
                .hidden_act
                .map(|v| serde_json::from_str::<Activation>(&v).unwrap_or(default.hidden_act))
                .unwrap_or(default.hidden_act),
            max_position_embeddings: value
                .max_position_embeddings
                .unwrap_or(default.max_position_embeddings),
            rms_norm_eps: value.rms_norm_eps.unwrap_or(default.rms_norm_eps),
            rope_theta: value.rope_theta as f64,
            sliding_window: value.sliding_window.or(default.sliding_window),
            use_flash_attn: default.use_flash_attn,
            head_dim: default.head_dim,
        }
    }
}
impl From<LLMConfig> for stable_lm::Config {
    // cannot fillin private field (use deserialize)
    fn from(_value: LLMConfig) -> Self {
        // default not exists
        let default = stable_lm::Config::stablelm_3b_4e1t(_value.use_flash_attn.unwrap_or(true));
        // XXX cannot init stable_lm::Config (field scope "crate")
        tracing::debug!("load stable_lm 3b 4e1t model params: {:?}", default);
        default
    }
}
// for deserialized stable_lm::Config to LLMConfigWrapper
impl From<stable_lm::Config> for LLMConfigWrapper {
    fn from(value: stable_lm::Config) -> Self {
        LLMConfigWrapper::StableLM(value)
    }
}

// TODO Quantized definitions
#[derive(Debug)]
pub enum LLMConfigWrapper {
    Falcon(falcon::Config),
    Llama(llama::Config),
    Mistral(mistral::Config),
    StableLM(stable_lm::Config),
    // QStableLM(quantized_stable_lm::Config),
}
impl Clone for LLMConfigWrapper {
    fn clone(&self) -> Self {
        match self {
            Self::Falcon(v) => Self::Falcon(falcon::Config {
                // XXX not implement Clone for falcon::Config
                vocab_size: v.vocab_size,
                hidden_size: v.hidden_size,
                num_hidden_layers: v.num_hidden_layers,
                num_attention_heads: v.num_attention_heads,
                layer_norm_epsilon: v.layer_norm_epsilon,
                initializer_range: v.initializer_range,
                use_cache: v.use_cache,
                bos_token_id: v.bos_token_id,
                eos_token_id: v.eos_token_id,
                hidden_dropout: v.hidden_dropout,
                attention_dropout: v.attention_dropout,
                n_head_kv: v.n_head_kv,
                alibi: v.alibi,
                new_decoder_architecture: v.new_decoder_architecture,
                multi_query: v.multi_query,
                parallel_attn: v.parallel_attn,
                bias: v.bias,
            }),
            Self::Llama(v) => Self::Llama(v.clone()),
            Self::Mistral(v) => Self::Mistral(v.clone()),
            Self::StableLM(v) => Self::StableLM(v.clone()),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        match (self, source) {
            (Self::Falcon(dest), Self::Falcon(src)) => {
                dest.vocab_size = src.vocab_size;
                dest.hidden_size = src.hidden_size;
                dest.num_hidden_layers = src.num_hidden_layers;
                dest.num_attention_heads = src.num_attention_heads;
                dest.layer_norm_epsilon = src.layer_norm_epsilon;
                dest.initializer_range = src.initializer_range;
                dest.use_cache = src.use_cache;
                dest.bos_token_id = src.bos_token_id;
                dest.eos_token_id = src.eos_token_id;
                dest.hidden_dropout = src.hidden_dropout;
                dest.attention_dropout = src.attention_dropout;
                dest.n_head_kv = src.n_head_kv;
                dest.alibi = src.alibi;
                dest.new_decoder_architecture = src.new_decoder_architecture;
                dest.multi_query = src.multi_query;
                dest.parallel_attn = src.parallel_attn;
                dest.bias = src.bias;
            }
            (Self::Llama(dest), Self::Llama(src)) => {
                *dest = src.clone();
            }
            (Self::Mistral(dest), Self::Mistral(src)) => {
                *dest = src.clone();
            }
            (Self::StableLM(dest), Self::StableLM(src)) => {
                *dest = src.clone();
            }
            (_, _) => panic!("model type mismatch"),
        }
    }
}

impl From<LLMConfig> for LLMConfigWrapper {
    fn from(value: LLMConfig) -> Self {
        match value.model_type() {
            Some(ModelType::Falcon) => Self::Falcon(value.into()),
            Some(ModelType::Llama) => Self::Llama(value.into()),
            Some(ModelType::Mistral) => Self::Mistral(value.into()),
            Some(ModelType::StableLM) => Self::StableLM(value.into()),
            None => {
                tracing::error!("unknown model type: {:?}", &value.architectures);
                unreachable!()
            }
        }
    }
}
