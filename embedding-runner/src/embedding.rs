use crate::model::BertLoaderImpl;
use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_wrapper::{bert::BertWrapper, models::ModelWrapper};
use command_utils::text::{SentenceSplitter, SentenceSplitterCreator};
use command_utils::util::result::FlatMap;
use itertools::Itertools;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

pub struct SentenceEmbedder {
    /// The model to embed sentences.(bert)
    pub model: BertWrapper,
    pub tokenizer: Tokenizer,
    pub device: Device,
    /// sentence segmenter
    pub splitter: SentenceSplitter,
    /// normalize embeddings vector
    pub normalize_embeddings: bool,
    /// prefix for each sentence in embedding (override env value. e.g. "query: ", "passage: " for multilingual-e5 model)
    pub prefix: Option<String>,
}

impl SentenceEmbedder {
    // max input length for bert (max_position_embeddings)
    const DEFAULT_MAX_LENGTH: usize = 512;
    pub fn new() -> Result<Self> {
        let model_loader = BertLoaderImpl::from_env()?;
        Self::new_from(&model_loader)
    }
    pub fn new_from(model_loader: &BertLoaderImpl) -> Result<Self> {
        // load model to gpu
        let wrapper = model_loader.load()?;

        let mut splitter_config = SentenceSplitterCreator::new_by_env()?;
        splitter_config.max_buf_length = Some(
            Self::DEFAULT_MAX_LENGTH - model_loader.prefix.as_ref().map(|p| p.len()).unwrap_or(0),
        );
        let model = match wrapper.model {
            // TODO from config (private field in candle...)
            ModelWrapper::Bert(m) => Ok(m),
            m => Err(anyhow!("cannot apply this method to model: {:?}", m)),
        }?;
        Ok(Self {
            model,
            tokenizer: wrapper.tokenizer,
            device: wrapper.device,
            splitter: splitter_config.create()?,
            normalize_embeddings: model_loader.normalize_embeddings,
            prefix: model_loader.prefix.clone(),
        })
    }

    fn split_with_normalize(&self, text: String, prefix: Option<&String>) -> Vec<String> {
        // Split text into sentences and normalize them.
        self.splitter
            .split(text)
            .into_iter()
            .map(|s| {
                let s = s.trim();
                prefix
                    .or(self.prefix.as_ref())
                    .map(|p| format!("{}{}", p, s))
                    .unwrap_or(s.to_string())
            })
            .collect()
    }

    pub fn generate_embeddings_to_vec(
        &mut self,
        article: String,
        prefix: Option<&String>,
    ) -> Result<Vec<Vec<f32>>> {
        self.generate_embeddings(article, prefix).map(|x| {
            x.into_iter()
                .flat_map(Self::embedding_tensor_to_vec)
                .collect_vec()
        })
    }
    /// Embed a batch of sentences
    pub fn generate_embeddings(
        &mut self,
        article: String,
        prefix: Option<&String>,
    ) -> Result<Vec<Tensor>> {
        let sentences = self.split_with_normalize(article, prefix);
        tracing::debug!("normalized sentences: {:?}", sentences);
        Ok(sentences
            .chunks(100) // in memory batch size for input sentences
            .flat_map(|batch| self.generate_embeddings_inner(batch))
            .flatten()
            .collect_vec())
    }

    fn generate_embeddings_inner(&mut self, sentences: &[String]) -> Result<Vec<Tensor>> {
        // https://github.com/huggingface/candle/blob/main/candle-examples/examples/bert/main.rs#L146
        let n_sentences = sentences.len();
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }
        let tokens = self
            .tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(anyhow::Error::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let embeddings = self.model.forward(&token_ids, Some(&token_type_ids), 0)?;
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = if self.normalize_embeddings {
            Self::normalize_l2(&embeddings)?
        } else {
            embeddings
        };
        let embeddings = embeddings.chunk(n_sentences, 0)?;

        Ok(embeddings)
    }

    pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }

    // sentence vector is 2-dim tensor but have only 1 rows (with avg pooling through sentence tokens above)
    pub fn embedding_tensor_to_vec(embedding: Tensor) -> Result<Vec<f32>> {
        embedding
            .get_on_dim(0, 0)
            .map_err(|e| e.into())
            .flat_map(|m| m.to_vec1().map_err(|e| e.into()))
    }
}

// test for SentenceEmbedder
#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;

    static E5SMALL_LOADER: Lazy<BertLoaderImpl> = Lazy::new(|| BertLoaderImpl {
        use_cpu: true,
        model_id: "intfloat/multilingual-e5-small".to_string(),
        normalize_embeddings: true,
        approximate_gelu: true,
        prefix: "query: ".to_string().into(),
    });

    #[test]
    fn test_split_with_normalize() {
        let embedder = SentenceEmbedder::new_from(&E5SMALL_LOADER).unwrap();
        let text = "これはテストです。それはテストではないです。http://localhost.org\n「あ。え。」"
            .to_string();
        let res = embedder.split_with_normalize(text, None);
        let prefix = E5SMALL_LOADER.prefix.as_ref().unwrap().clone();
        assert_eq!(
            res,
            vec![
                format!("{}これはテストです。", prefix),
                format!("{}それはテストではないです。", prefix),
                format!("{}http://localhost.org", prefix),
                format!("{}「あ。え。」", prefix),
            ],
        );
    }

    #[test]
    fn test_embedding_tensor_to_vec() {
        let mut embedder = SentenceEmbedder::new_from(&E5SMALL_LOADER).expect("load embedded");
        let text = "これはテストです。".to_string();
        let res = embedder
            .generate_embeddings(text, Some(&"query:".to_string()))
            .unwrap();
        assert_eq!(res.len(), 1);
        let res1 = SentenceEmbedder::embedding_tensor_to_vec(res[0].clone()).unwrap();
        assert_eq!(res1.len(), 384);
        assert!(
            (res[0]
                .sqr()
                .unwrap()
                .sum_keepdim(1)
                .unwrap()
                .sqrt() // L2 norm
                .unwrap()
                .to_vec2::<f32>() // 2-dim tensor to vec
                .unwrap()
                .into_iter()
                .flatten()
                .collect_vec()[0]
                - 1.0) // normalized L2 norm
                .abs()
                < 0.00001 // equals
        );
    }
}
