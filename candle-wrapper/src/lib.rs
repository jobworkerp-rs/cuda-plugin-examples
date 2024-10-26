pub mod bert;
pub mod llm;
pub mod loader;
pub mod models;
pub mod whisper;

use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait PluginRunner: Send + Sync {
    fn name(&self) -> String;
    fn load(&mut self, operation: Vec<u8>) -> Result<()>;
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>>;
    fn cancel(&self) -> bool;
    fn operation_proto(&self) -> String;
    fn job_args_proto(&self) -> String;
    fn result_output_proto(&self) -> Option<String>;
    // if true, use job result of before job, else use job args from request
    fn use_job_result(&self) -> bool;
}
