pub mod audio;
pub mod runner;
pub mod utils;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use prost::Message;
use protobuf::whisper::WhisperSegment;
use runner::WhisperRunModel;
use std::{alloc::System, fs::File, io::Cursor, path::PathBuf};
use tempfile::{Builder, TempDir};

use crate::runner::{whisper::Task, WhisperParams};

pub mod protobuf {
    pub mod whisper {
        include!(concat!(env!("OUT_DIR"), "/whisper.rs"));
    }
}

#[global_allocator]
static ALLOCATOR: System = System;

#[async_trait]
pub trait PluginRunner: Send + Sync {
    fn name(&self) -> String;
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>>;
    fn cancel(&self) -> bool;
}

// suppress warn improper_ctypes_definitions
#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn load_plugin() -> Box<dyn PluginRunner + Send + Sync> {
    Box::new(WhisperRunnerPlugin {
        model: WhisperRunModel::new().unwrap(),
    })
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn free_plugin(ptr: Box<dyn PluginRunner + Send + Sync>) {
    drop(ptr);
}

pub struct WhisperRunnerPlugin {
    pub model: WhisperRunModel,
}
// static DATA: OnceCell<Bytes> = OnceCell::new();

impl WhisperRunnerPlugin {
    pub fn new() -> Result<Self> {
        let model = WhisperRunModel::new()?;
        Ok(Self { model })
    }
    // XXX unused
    async fn get_temp_file(&self, url: &String) -> Result<(TempDir, PathBuf)> {
        // The temporary directory is automatically removed on program exit.
        let tmp_dir = Builder::new().prefix("whisper").tempdir()?;
        let response = reqwest::get(url).await?;

        let fname = response
            .url()
            .path_segments()
            .and_then(|segments| segments.last())
            .and_then(|name| if name.is_empty() { None } else { Some(name) })
            .unwrap_or("tmp.bin");
        let fname = tmp_dir.path().join(fname);
        tracing::info!("download file will be located under: '{:?}'", &fname);
        let mut dest = File::create(&fname)?;
        let content = response.bytes().await?;
        std::io::copy(&mut content.as_ref(), &mut dest)?;
        Ok((tmp_dir, fname))
    }
}

impl PluginRunner for WhisperRunnerPlugin {
    fn name(&self) -> String {
        // specify as same string as worker.operation
        String::from("WhisperRunner")
    }
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>> {
        // let args = serde_json::from_slice::<WhisperArgs>(&arg)?;
        let args = protobuf::whisper::WhisperRequest::decode(&mut Cursor::new(arg))
            .map_err(|e| anyhow!("decode error: {}", e))?;

        let start = std::time::Instant::now();
        // for async function
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let (_tmp_dir, path) = if args.url.trim().starts_with("http") {
                    tracing::info!("download file from url: {}", &args.url);
                    self.get_temp_file(&args.url).await?
                } else {
                    tracing::info!("decode file: {}", &args.url);
                    (TempDir::new()?, PathBuf::from(&args.url))
                };
                let language = match args.lang() {
                    protobuf::whisper::Language::Auto => None,
                    l => Some(l.as_str_name().to_uppercase().to_string()),
                };

                let task = match args.task() {
                    protobuf::whisper::Task::Transcribe => Task::Transcribe, // default
                    protobuf::whisper::Task::Translate => Task::Translate,
                };

                let params = WhisperParams {
                    seed: args.seed.unwrap_or(rand::random::<u64>()),
                    language,
                    task,
                    timestamps: args.timestamps,
                };
                let segs = self
                    .model
                    .decode_file(path.to_string_lossy().as_ref(), params)?;
                tracing::info!(
                    "END OF WhisperRunner: elapsed seconds: {}",
                    start.elapsed().as_secs_f64()
                );
                let bin = protobuf::whisper::WhisperResponse {
                    segments: segs
                        .into_iter()
                        .map(|s| WhisperSegment {
                            start: s.start,
                            duration: s.duration,
                            text: s.dr.text,
                        })
                        .collect(),
                };
                let mut buf = Vec::with_capacity(bin.encoded_len());
                bin.encode(&mut buf).unwrap();
                Ok(vec![buf])
            })
    }
    fn cancel(&self) -> bool {
        tracing::warn!("WhisperRunner cancel: not implemented!");
        false
    }
}
pub struct WhisperSegmentWriter {}

// XXX same as Segment impl
impl WhisperSegmentWriter {
    fn vtt_header() -> String {
        "WEBVTT\n\n".to_owned()
    }
    pub fn build_vtt(segs: &Vec<WhisperSegment>) -> String {
        let mut vtt = Self::vtt_header();
        for seg in segs {
            vtt.push_str(&Self::vtt_cue(seg));
        }
        vtt
    }
    //extract floating number 3 chars as string from f64
    fn floating_num_str(num: f64) -> String {
        let mut num_str = format!("{:.03}", num);
        // trim before floating point
        num_str = num_str
            .trim_start_matches(['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            .to_string();
        num_str
    }
    fn start_hms(s: &WhisperSegment) -> String {
        let secs = s.start as u32;
        let mins = secs / 60;
        let secs = secs % 60;
        let hours = mins / 60;
        let mins = mins % 60;
        format!(
            "{:02}:{:02}:{:02}{}",
            hours,
            mins,
            secs,
            Self::floating_num_str(s.start)
        )
    }
    fn end_hms(s: &WhisperSegment) -> String {
        let end = s.start + s.duration;
        let secs = (end) as u32;
        let mins = secs / 60;
        let secs = secs % 60;
        let hours = mins / 60;
        let mins = mins % 60;
        format!(
            "{:02}:{:02}:{:02}{}",
            hours,
            mins,
            secs,
            Self::floating_num_str(end)
        )
    }
    // https://www.w3.org/TR/webvtt1/
    fn vtt_cue(s: &WhisperSegment) -> String {
        let start = Self::start_hms(s);
        let end = Self::end_hms(s);
        let text = &s.text;
        format!("{} --> {}\n{}\n\n", start, end, text)
    }
}
