use std::io::Cursor;

use candle_wrapper::PluginRunner;
use clap::Parser;
use llm_runner::{protobuf::llm::CandleLlmArg, LlmRunnerPlugin};
use prost::Message;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 1000)]
    sample_len: u32,

    /// The initial prompt.
    #[arg(long)]
    prompt: String,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.2)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Only print request payload as base64 encodings (for jobworkerp request)
    #[arg(long, default_value = "false")]
    request_output_only: bool,
}

impl Args {
    fn new() -> Self {
        Self::parse()
    }
    fn to_request(&self) -> CandleLlmArg {
        CandleLlmArg {
            prompt: self.prompt.clone(),
            sample_len: self.sample_len,
            temperature: self.temperature,
            top_p: self.top_p,
            repeat_penalty: Some(self.repeat_penalty),
            repeat_last_n: Some(self.repeat_last_n as u32),
            seed: self.seed,
            need_print: true, // for cli
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt().with_env_filter("info").init();
    let args = Args::new();
    tracing::debug!("args:{:?}", args);
    let request = args.to_request();
    tracing::debug!("parameters:{:?}", request);

    let mut buf = Vec::with_capacity(request.encoded_len());
    request.encode(&mut buf).unwrap();

    if args.request_output_only {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        println!("request base64: {}", STANDARD.encode(&buf));
    } else {
        let mut plugin = LlmRunnerPlugin::new();
        plugin.load_config_from_env()?;
        let result = plugin.run(buf)?;
        let text = if let Some(res) = result.first() {
            String::from_utf8_lossy(res).to_string()
        } else {
            tracing::warn!("no result");
            "".to_string()
        };
        tracing::info!("response: {}", text);
    }
    Ok(())
}
