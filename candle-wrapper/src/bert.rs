use anyhow::{anyhow, Result};
use candle_core::Tensor;
use candle_transformers::models::bert::BertModel;
use debug_stub_derive::DebugStub;

#[derive(DebugStub)]
pub enum BertWrapper {
    Bert(#[debug_stub = "BertModel"] BertModel),
}

// for Bert model
impl BertWrapper {
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        _seqlen_offset: usize,
    ) -> Result<Tensor> {
        match (self, token_type_ids) {
            (Self::Bert(m), Some(ttids)) => m
                .forward(input_ids, ttids, attention_mask)
                .map_err(|e| e.into()),
            (Self::Bert(_m), None) => Err(anyhow!("token type ids is necessary for bert model",)),
        }
    }
}
