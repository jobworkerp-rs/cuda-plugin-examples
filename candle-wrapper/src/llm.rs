pub mod config;
pub mod error;

use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::falcon;
use candle_transformers::models::llama;
use candle_transformers::models::mistral;
use candle_transformers::models::stable_lm;

#[derive(Debug)]
pub enum LLMModelWrapper {
    Falcon(falcon::Falcon),
    Llama(llama::Llama, llama::Cache),
    Mistral(mistral::Model),
    StableLM(stable_lm::Model),
    // QStableLM(quantized_stable_lm::Model),
}
impl LLMModelWrapper {
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        match self {
            // TODO not working?
            Self::Falcon(m) => m.forward(input_ids).map_err(|e| e.into()),
            Self::Llama(m, c) => m.forward(input_ids, seqlen_offset, c).map_err(|e| e.into()),
            Self::Mistral(m) => m.forward(input_ids, seqlen_offset).map_err(|e| e.into()),
            Self::StableLM(m) => m.forward(input_ids, seqlen_offset).map_err(|e| e.into()),
            // Self::QStableLM(_) => todo!(),
        }
    }
}
