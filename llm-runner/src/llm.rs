pub mod model;
pub mod tokenizer;

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;
//
// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

use crate::llm::model::LLM;
use crate::protobuf::llm::InferenceRequest;
use anyhow::{Error as E, Result};
use candle_core::{DType, Tensor};
use candle_transformers::generation::LogitsProcessor;
use serde::{Deserialize, Serialize};

use self::model::{LLMModelLoader, LLMModelLoaderConfig};

// args for inference
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InferenceArgs {
    /// The initial prompt.
    pub prompt: String,

    /// The length of the sample to generate (in tokens).
    pub sample_len: usize,

    /// The temperature used to generate samples.
    pub temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    pub top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    pub repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    pub repeat_last_n: usize,

    /// The seed to use when generating random samples.
    pub seed: u64,

    /// false for plugin runner, true for cli
    #[serde(default)]
    pub use_print: bool,
}

impl Default for InferenceArgs {
    fn default() -> Self {
        Self {
            prompt: "".to_string(),
            sample_len: 100,
            temperature: None,
            top_p: None,
            repeat_penalty: 1.,
            repeat_last_n: 1,
            seed: 42,
            use_print: false,
        }
    }
}
impl From<InferenceRequest> for InferenceArgs {
    fn from(req: InferenceRequest) -> Self {
        Self {
            prompt: req.prompt,
            sample_len: req.sample_len as usize,
            temperature: req.temperature,
            top_p: req.top_p,
            repeat_penalty: req.repeat_penalty.unwrap_or(1.2),
            repeat_last_n: req.repeat_last_n.unwrap_or(64) as usize,
            seed: req.seed.unwrap_or(rand::random::<u64>()).wrapping_add(42),
            use_print: req.need_print,
        }
    }
}

pub struct LlmInferencer {
    model_loader: LLMModelLoader,
    llm: LLM,
}

impl LlmInferencer {
    pub fn new(config: LLMModelLoaderConfig) -> Result<Self> {
        let model_loader = LLMModelLoader::new(config);
        let llm = model_loader.load()?;
        Ok(Self { model_loader, llm })
    }
    pub fn generate_text(&mut self, args: InferenceArgs) -> Result<String> {
        use scopeguard::guard;
        use std::io::Write;
        let use_print = args.use_print;
        let mut tokenizer = guard(&mut self.llm.tokenizer, |t| {
            // clear on return
            tracing::debug!("tokenizer cleared on end");
            t.clear();
        });
        let use_kv_cache = true; // TODO configurable
        let eos_token_id = tokenizer.get_token(self.model_loader.config.eos_token.as_str());

        let prompt = args.prompt;
        let mut tokens = tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut output = String::new();
        for &t in tokens.iter() {
            if let Some(t) = tokenizer.next_token(t)? {
                if use_print {
                    print!("{}", &t);
                    std::io::stdout().flush()?;
                }
                output.push_str(t.as_str());
            }
        }
        tracing::debug!("starting the inference loop");
        let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);
        let start_gen = std::time::Instant::now();
        // let mut index_pos = 0;
        let mut token_generated = 0;
        for index in 0..args.sample_len {
            let context_size = if use_kv_cache && index > 0 {
                1
            } else {
                tokens.len()
            };
            let start_pos: usize = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.llm.device)?.unsqueeze(0)?;
            //let logits = (&mut self.llm.model).forward(&input, index_pos)?;
            let logits = self.llm.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            // let logits = logits.squeeze(0)?;
            let logits = if args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    args.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            // index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)?;
            token_generated += 1;
            tokens.push(next_token);

            // Extracting the last token as a string is complicated, here we just apply some simple
            // heuristics as it seems to work well enough for this example. See the following for more
            // details:
            // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
            if let Some(text) = tokenizer.next_token(next_token)? {
                // let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                if use_print {
                    print!("{}", &text.as_str());
                    std::io::stdout().flush()?;
                }
                output.push_str(&text);
            }
            if Some(next_token) == eos_token_id {
                break;
            }
        }
        if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
            if use_print {
                print!("{}", &rest.as_str());
                std::io::stdout().flush()?;
            }
            output.push_str(&rest);
            // print!("{rest}");
            // std::io::stdout().flush()?;
        }

        let dt = start_gen.elapsed();
        tracing::info!(
            "\n\n{} tokens generated ({} token/s)\n",
            token_generated,
            token_generated as f64 / dt.as_secs_f64(),
        );
        Ok(output)
    }
}
