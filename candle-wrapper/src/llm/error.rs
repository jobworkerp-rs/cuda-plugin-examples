use thiserror::Error;

#[derive(Error, Debug)]
pub enum LLMErrors {
    #[error("failed to load plugin")]
    LoadPluginError,
    #[error("cannot read model config from env:")]
    EnvError(#[from] envy::Error),
    #[error("failed to load config")]
    LoadConfigError,
    #[error("model file(s) not found")]
    ModelNotFound,
    #[error("failed to run")]
    RunError,
}
