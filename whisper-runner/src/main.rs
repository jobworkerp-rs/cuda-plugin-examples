use std::{io::Cursor, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use command_utils::util::{option::FlatMap as OptionFlatMap, result::FlatMap};
use prost::Message;
use serde::Deserialize;
use url::Url;
use whisper_runner::{
    protobuf::whisper::WhisperSegment, runner::whisper::Task, PluginRunner, WhisperRunnerPlugin,
    WhisperSegmentWriter,
};

#[derive(Parser, Debug, Deserialize, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// input local file path or http(s) url
    #[arg(long)]
    pub input: String,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,

    /// Language.
    #[arg(long)]
    pub language: Option<String>,

    /// Task, when no task is specified, the input tokens contain only the sot token which can
    /// improve things when in no-timestamp mode.
    #[arg(long, default_value = "transcribe")]
    pub task: Task,

    /// Timestamps mode
    #[arg(long, default_value = "false")]
    pub timestamps: bool,

    /// Only print request payload as base64 encodings (for jobworkerp request)
    #[arg(long, default_value = "false")]
    request_output_only: bool,
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    // XXX blocking
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move { command_utils::util::tracing::tracing_init_from_env().await })?;

    let args = Args::parse();
    if args.request_output_only {
        let req = whisper_runner::protobuf::whisper::WhisperRequest {
            url: args.input,
            lang: whisper_runner::protobuf::whisper::Language::Auto as i32,
            task: if args.task == Task::Transcribe { 0 } else { 1 },
            timestamps: args.timestamps,
            seed: Some(args.seed),
        };
        let mut buf = Vec::with_capacity(req.encoded_len());
        req.encode(&mut buf).unwrap();

        use base64::{engine::general_purpose::STANDARD, Engine as _};
        println!("request base64: {}", STANDARD.encode(&buf));
    } else {
        let segs = decode_file(
            &args.input,
            args.task,
            args.language.as_ref(),
            args.timestamps,
            args.seed,
        )?;
        save_segs(&segs, &args.input)?;
    }
    Ok(())
}

fn decode_file(
    path: &str,
    task: Task,
    lang: Option<&String>,
    timestamps: bool,
    seed: u64,
) -> Result<Vec<WhisperSegment>> {
    let mut runner = WhisperRunnerPlugin::new()?;
    let lang = lang
        .flat_map(|l| whisper_runner::protobuf::whisper::Language::from_str_name(l))
        .unwrap_or(whisper_runner::protobuf::whisper::Language::Auto);
    let req = whisper_runner::protobuf::whisper::WhisperRequest {
        url: path.to_owned(),
        lang: lang as i32,
        task: if task == Task::Transcribe { 0 } else { 1 },
        timestamps,
        seed: Some(seed),
    };
    let mut buf = Vec::with_capacity(req.encoded_len());
    req.encode(&mut buf).unwrap();

    let result = runner.run(buf)?;
    if let Some(res) = result.first() {
        let res =
            whisper_runner::protobuf::whisper::WhisperResponse::decode(&mut Cursor::new(res))?
                .segments;

        tracing::debug!("result:len={},  data={:?}", res.len(), res);

        Ok(res)
    } else {
        Ok(vec![])
    }
}

fn save_segs(segs: &Vec<WhisperSegment>, path: &String) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(vtt_filename_from_url_or_file_path(path))?;
    file.write_all(WhisperSegmentWriter::build_vtt(segs).as_bytes())?;
    Ok(())
}

fn vtt_filename_from_url_or_file_path(path: &String) -> String {
    let name = if path.starts_with("http") {
        Url::parse(path).context("url parse").flat_map(|u| {
            u.path_segments()
                .flat_map(|s| s.last().map(|s| s.to_owned()))
                .ok_or(anyhow!("cannot get path"))
        })
    } else {
        Ok(PathBuf::from(path)
            .file_name()
            .flat_map(|s| s.to_str())
            .map(|s| s.to_owned())
            .unwrap_or("output".to_owned()))
    }
    .unwrap_or("output".to_string());
    format!(
        "{}.vtt",
        if name.is_empty() {
            "output".to_string()
        } else {
            name
        }
    )
}

// create test for vtt_filename_from_url_or_file_path()
#[test]
fn test_vtt_filename_from_url_or_file_path() {
    assert_eq!(
        vtt_filename_from_url_or_file_path(&"https://example.com/".to_string()),
        "output.vtt"
    );
    assert_eq!(
        vtt_filename_from_url_or_file_path(&"https://example.com/foo".to_string()),
        "foo.vtt"
    );
    assert_eq!(
        vtt_filename_from_url_or_file_path(&"https://example.com/foo/".to_string()),
        "output.vtt"
    );
    assert_eq!(
        vtt_filename_from_url_or_file_path(&"https://example.com/foo.txt".to_string()),
        "foo.txt.vtt"
    );
    assert_eq!(
        vtt_filename_from_url_or_file_path(&"https://example.com/foo.txt?query".to_string()),
        "foo.txt.vtt"
    );
    assert_eq!(
        vtt_filename_from_url_or_file_path(&"https://example.com/foo.txt#fragment".to_string()),
        "foo.txt.vtt"
    );
    assert_eq!(
        vtt_filename_from_url_or_file_path(
            &"https://example.com/bar/foo.txt?query#fragment".to_string()
        ),
        "foo.txt.vtt"
    );
    assert_eq!(
        vtt_filename_from_url_or_file_path(&"./dir/foo.txt".to_string()),
        "foo.txt.vtt"
    );
}
