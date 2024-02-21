use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::Device;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::models::{self, ConfigWrapper, HFModelSetting, ModelWrapper, Models};

// TODO using type matching
pub trait HFModelLoader {
    // GPU or CPU (GPU is only available for cuda id:0 device)
    fn device(cpu: bool) -> Result<Device> {
        if cpu {
            Ok(Device::Cpu)
        } else if cuda_is_available() {
            Ok(Device::new_cuda(0)?)
        } else if metal_is_available() {
            Ok(Device::new_metal(0)?)
        } else {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                tracing::info!(
                    "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
                );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                tracing::info!(
                    "Running on CPU, to run on GPU, build this example with `--features cuda`"
                );
            }
            Ok(Device::Cpu)
        }
    }
    // // TODO resolve from model_config
    // fn dtype(&self, device: &Device) -> DType;
    /// load config using inner load()
    fn load_config_inner(&self, config_file: &Path) -> Result<ConfigWrapper>;

    /// load model using inner load()
    // TODO select model from config
    // TODO use proper load method by weight files extension
    fn load_model_inner(
        &self,
        config: &ConfigWrapper,
        weight_files: &[PathBuf],
    ) -> Result<(ModelWrapper, Device)>;

    // devide each model to each files
    fn load_from_setting(&self, setting: HFModelSetting) -> Result<Models> {
        let tokenizer_filename = std::path::PathBuf::from(
            setting
                .tokenizer_file
                .as_deref()
                .unwrap_or(models::TOKENIZER_FILE),
        );
        let mfile = std::path::PathBuf::from(
            setting
                .model_config_file
                .as_deref()
                .unwrap_or(models::MODEL_CONFIG_FILE),
        );

        // TODO error handling
        let filenames = setting
            .weight_files
            .as_ref()
            .unwrap_or(
                &models::WEIGHT_FILES_SAFETENSOR_SINGLE
                    .map(|s| s.to_string())
                    .as_ref()
                    .to_vec(),
            )
            .iter()
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>();
        // load model from hf or not
        let (tf, mf, wf) = if let Some(mid) = setting.model_id.as_ref() {
            tracing::info!("load model with hugginface: {}", &mid);
            Self::load_files_from_repository(
                mid,
                setting.revision.as_ref(),
                &tokenizer_filename,
                &mfile,
                &filenames,
            )
        } else {
            tracing::info!("load model with localfile: {}", &mfile.to_string_lossy());
            Ok((tokenizer_filename, mfile, filenames))
        }?;

        self.load_files(&tf, &mf, &wf)
    }

    // load files from hf repository
    fn load_files_from_repository(
        model_id: &str,
        revision: Option<&String>,
        tokenizer_filename: &Path,
        model_filename: &Path,
        weight_filenames: &[PathBuf],
    ) -> Result<(PathBuf, PathBuf, Vec<PathBuf>)> {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_owned(),
            RepoType::Model,
            revision
                .unwrap_or(&models::MAIN_REVISION.to_string())
                .clone(), // XXX default revision
        ));
        let tokenizer_filename = repo.get(&tokenizer_filename.as_os_str().to_string_lossy())?;
        // TODO error handling
        let filenames = weight_filenames
            .iter()
            .flat_map(|f| repo.get(&f.as_os_str().to_string_lossy()))
            .collect::<Vec<_>>();
        // path got from repository
        let mfile = repo.get(&model_filename.as_os_str().to_string_lossy())?;

        Ok((tokenizer_filename, mfile, filenames))
    }

    fn load_files(
        &self,
        tokenizer_filename: &Path,
        model_filename: &Path,
        weight_filenames: &[PathBuf],
    ) -> Result<Models> {
        tracing::debug!("load tokenizer: {:?}", &tokenizer_filename);
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| {
            anyhow::anyhow!("error in load tokenizer {:?}: {:?}", &tokenizer_filename, e)
        })?;

        // path got from repository
        tracing::debug!("load model file: {:?}", &model_filename);
        let config = self.load_config_inner(model_filename)?;
        tracing::debug!("load weight file: {:?}", &weight_filenames);
        let (model, device) = self.load_model_inner(&config, weight_filenames)?;

        Ok(Models {
            model,
            tokenizer,
            device,
        })
    }
}
