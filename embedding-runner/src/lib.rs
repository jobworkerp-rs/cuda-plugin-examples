pub mod embedding;
pub mod jobworkerp;
pub mod model;

use anyhow::{anyhow, Result};
use embedding::SentenceEmbedder;
use itertools::Itertools;
use prost::Message;
use std::io::Cursor;

pub mod protobuf {
    pub mod embedding {
        include!(concat!(env!("OUT_DIR"), "/embedding.rs"));
    }
}
pub trait PluginRunner: Send + Sync {
    fn name(&self) -> String;
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>>;
    fn cancel(&self) -> bool;
}

// plugin entry point
#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn load_plugin() -> Box<dyn PluginRunner + Send + Sync> {
    dotenvy::dotenv().ok();
    let plugin = SentenceBertRunnerPlugin::new().expect("failed to load plugin");
    Box::new(plugin)
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn free_plugin(ptr: Box<dyn PluginRunner + Send + Sync>) {
    drop(ptr);
}

pub struct SentenceBertRunnerPlugin {
    embedder: SentenceEmbedder,
}
// static DATA: OnceCell<Bytes> = OnceCell::new();

impl SentenceBertRunnerPlugin {
    const RUNNER_NAME: &'static str = "SentenceBertRunner";
    pub fn new() -> Result<Self> {
        let embedder = SentenceEmbedder::new()?;
        Ok(Self { embedder })
    }
}

// arg: Vec<u8> <- text: String
// return: Vec<u8> <- embeddings: Vec<Vec<f32>> (bincode)
impl PluginRunner for SentenceBertRunnerPlugin {
    fn name(&self) -> String {
        // specify as same string as worker.operation
        String::from(Self::RUNNER_NAME)
    }
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>> {
        let req = protobuf::embedding::SentenceEmbeddingRequest::decode(&mut Cursor::new(arg))
            .map_err(|e| anyhow!("decode error: {}", e))?;
        let text = req.article;
        let prefix = req.prefix;
        let emb = self
            .embedder
            .generate_embeddings(text.to_string(), prefix.as_ref())?
            .into_iter()
            .flat_map(|v| {
                SentenceEmbedder::embedding_tensor_to_vec(v)
                    .map(|em| protobuf::embedding::SentenceEmbeddingVector { vector: em })
            })
            .collect_vec();
        let bin = protobuf::embedding::SentenceEmbeddingResponse { embeddings: emb };
        let mut buf = Vec::with_capacity(bin.encoded_len());
        bin.encode(&mut buf).unwrap();
        // let bin = bincode::serialize(&emb)?;
        Ok(vec![buf])
    }
    fn cancel(&self) -> bool {
        tracing::warn!("EmbeddingSentenceRunner: cancel not implemented!");
        false
    }
}
