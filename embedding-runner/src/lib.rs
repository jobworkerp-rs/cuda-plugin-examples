pub mod embedding;
pub mod jobworkerp;
pub mod model;

use anyhow::{anyhow, Result};
use candle_wrapper::PluginRunner;
use embedding::SentenceEmbedder;
use itertools::Itertools;
use model::BertLoaderImpl;
use prost::Message;
use protobuf::embedding::{EmbeddingArg, EmbeddingOperation, EmbeddingResult};
use std::io::Cursor;

pub mod protobuf {
    pub mod embedding {
        include!(concat!(env!("OUT_DIR"), "/embedding.rs"));
    }
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
    embedder: Option<SentenceEmbedder>,
}

impl SentenceBertRunnerPlugin {
    const RUNNER_NAME: &'static str = "embedding.Embedding";
    // worker operation data (need serializing to bytes)
    // const OPERATION: Lazy<WorkerOperation> = Lazy::new(|| WorkerOperation {
    //     operation: Some(Operation::Plugin(EmbeddingOperation {
    //         model_id: "intfloat/multilingual-e5-small".to_string(),
    //         use_cpu: false,
    //         normalize_embeddings: false,
    //         approximate_gelu: true,
    //         prefix: Some("query: ".to_string()),
    //     })),
    // });
    pub fn new() -> Result<Self> {
        Ok(Self { embedder: None })
    }
    pub fn load_model(&mut self, loader: &BertLoaderImpl) -> Result<()> {
        let embedder = SentenceEmbedder::new_from(loader)?;
        self.embedder = Some(embedder);
        Ok(())
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
        if let Some(embedder) = self.embedder.as_mut() {
            let req = EmbeddingArg::decode(&mut Cursor::new(arg))
                .map_err(|e| anyhow!("decode error: {}", e))?;
            let text = req.article;
            let prefix = req.prefix;
            let emb = embedder
                .generate_embeddings(text.to_string(), prefix.as_ref())?
                .into_iter()
                .flat_map(|v| {
                    SentenceEmbedder::embedding_tensor_to_vec(v).map(|em| {
                        protobuf::embedding::embedding_result::SentenceEmbeddingVector {
                            vector: em,
                        }
                    })
                })
                .collect_vec();
            let bin = EmbeddingResult { embeddings: emb };
            let mut buf = Vec::with_capacity(bin.encoded_len());
            bin.encode(&mut buf).unwrap();
            // let bin = bincode::serialize(&emb)?;
            Ok(vec![buf])
        } else {
            Err(anyhow!("embedder is not initialized"))
        }
    }
    fn cancel(&self) -> bool {
        tracing::warn!("EmbeddingSentenceRunner: cancel not implemented!");
        false
    }

    fn load(&mut self, operation: Vec<u8>) -> Result<()> {
        EmbeddingOperation::decode(&mut Cursor::new(operation))
            .map_err(|e| anyhow!("decode error: {}", e))
            .and_then(|op| self.load_model(&BertLoaderImpl::from(op)))?;
        Ok(())
    }

    fn operation_proto(&self) -> String {
        include_str!("../protobuf/embedding/embedding_operation.proto").to_string()
    }

    fn job_args_proto(&self) -> String {
        include_str!("../protobuf/embedding/embedding_arg.proto").to_string()
    }

    fn result_output_proto(&self) -> Option<String> {
        Some(include_str!("../protobuf/embedding/embedding_result.proto").to_string())
    }

    fn use_job_result(&self) -> bool {
        todo!()
    }
}
