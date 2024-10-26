use std::io::Cursor;

use candle_wrapper::PluginRunner;
use clap::Parser;
use embedding_runner::{
    model::BertLoaderImpl,
    protobuf::embedding::{EmbeddingArg, EmbeddingResult},
    SentenceBertRunnerPlugin,
};
use itertools::Itertools;
use prost::Message;
use tracing::Level;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable debug print
    #[arg(long, default_value = "false")]
    debug: bool,

    /// The sentences to embed
    #[arg(long)]
    sentences: String,

    /// The prefix to use sentence
    #[arg(long, default_value = "None")]
    prefix: Option<String>,

    /// Only print request payload as base64 encodings (for jobworkerp request)
    #[arg(long, default_value = "false")]
    request_output_only: bool,
}

// create embedding for all articles
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_max_level(if args.debug {
            Level::DEBUG
        } else {
            Level::INFO
        })
        .init();
    let req = EmbeddingArg {
        article: args.sentences,
        prefix: args.prefix,
    };
    let mut buf = Vec::with_capacity(req.encoded_len());
    req.encode(&mut buf).unwrap();

    // only print request payload as base64 encodings
    if args.request_output_only {
        use base64::{engine::general_purpose::STANDARD, Engine as _};

        println!("request base64: {}", STANDARD.encode(&buf));
    } else {
        let mut runner = SentenceBertRunnerPlugin::new()?;
        runner.load_model(&BertLoaderImpl::from_env()?)?;

        let res = runner.run(buf)?;
        let res = res.first().unwrap();
        let res = EmbeddingResult::decode(&mut Cursor::new(res))?.embeddings;

        tracing::info!(
            "Embedding vecs({},{})",
            res.len(),
            res.first().map(|e| e.vector.len()).unwrap_or(0),
        );
        // print result
        println!(
            "{}",
            serde_json::to_string(&res.into_iter().map(|v| v.vector).collect_vec())?
        );
    }
    Ok(())
}
