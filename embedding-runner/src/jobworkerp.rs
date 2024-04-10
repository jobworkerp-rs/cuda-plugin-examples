use anyhow::{anyhow, Result};
use jobworkerp_client::{
    client::JobworkerpClientImpl,
    jobworkerp::{
        data::{
            runner_arg::Data, JobResultData, PluginArg, QueueType, ResponseType, RetryPolicy,
            RetryType, RunnerArg, RunnerType, Worker, WorkerData,
        },
        service::{CreateWorkerRequest, JobRequest, WorkerNameRequest},
    },
};
use std::time::Duration;

use crate::SentenceBertRunnerPlugin;

pub struct JobworkerpEmbeddingClient {
    pub jobworkerp_client: JobworkerpClientImpl,
    pub channel_name: String,
}
impl JobworkerpEmbeddingClient {
    pub async fn new(timeout_sec: u32) -> Result<JobworkerpEmbeddingClient> {
        let jobworkerp_client = JobworkerpClientImpl::new(
            std::env::var("JOBWORKERP_ADDR").expect("JOBWORKERP_ADDR is not set"),
            Duration::from_secs(timeout_sec.into()),
        )
        .await?;

        let ch = std::env::var("JOBWORKERP_EMBEDDING_WORKER_CHANNEL").unwrap_or("gpu".to_string());
        Ok(JobworkerpEmbeddingClient {
            jobworkerp_client,
            channel_name: ch,
        })
    }
    pub async fn find_or_create_worker(&self) -> Result<Worker> {
        let policy = RetryPolicy {
            r#type: RetryType::Exponential as i32,
            interval: 3000,
            max_interval: 60000,
            max_retry: 3,
            basis: 2.0,
        };
        // TODO channel name etc from env
        let worker_data = WorkerData {
            name: "SentenceBertWorker".to_string(),
            r#type: RunnerType::Plugin as i32,
            operation: Some(SentenceBertRunnerPlugin::OPERATION.clone()),
            retry_policy: Some(policy.clone()),
            periodic_interval: 0,
            channel: Some(self.channel_name.clone()),
            queue_type: QueueType::Redis as i32,
            response_type: ResponseType::Direct as i32,
            store_success: false,
            store_failure: false,
            next_workers: vec![],
            use_static: false,
        };
        let mut worker_cli = self.jobworkerp_client.worker_client().await;

        let worker = worker_cli
            .find_by_name(WorkerNameRequest {
                name: worker_data.name.clone(),
            })
            .await?
            .into_inner()
            .data;

        // if not found, create sentence embedding worker
        let worker = if let Some(w) = worker {
            w
        } else {
            let wid = worker_cli
                .create(CreateWorkerRequest {
                    name: "SentenceBertWorker".to_string(),
                    operation: Some(SentenceBertRunnerPlugin::OPERATION.clone()),
                    retry_policy: Some(policy),
                    channel: Some(self.channel_name.clone()),
                    // queue_type: QueueType::Redis as i32, // default
                    response_type: Some(ResponseType::Direct as i32),
                    next_workers: vec![],
                    ..Default::default()
                })
                .await?
                .into_inner()
                .id
                .ok_or(anyhow!("create worker response is empty?"))?;
            Worker {
                id: Some(wid),
                data: Some(worker_data),
            }
        };
        Ok(worker)
    }
    pub async fn enqueue_embedding_job(
        &self,
        sentence: String,
        timeout_sec: u32,
    ) -> Result<JobResultData> {
        let worker = self.find_or_create_worker().await?;
        let mut job_cli = self.jobworkerp_client.job_client().await;
        job_cli
            .enqueue(JobRequest {
                arg: Some(RunnerArg {
                    data: Some(Data::Plugin(PluginArg {
                        arg: sentence.into_bytes(),
                    })),
                }),
                timeout: Some((timeout_sec * 1000).into()),
                worker: Some(
                    jobworkerp_client::jobworkerp::service::job_request::Worker::WorkerId(
                        worker.id.unwrap(),
                    ),
                ),
                ..Default::default()
            })
            .await?
            .into_inner()
            .result
            .ok_or(anyhow!("result not found"))?
            .data
            .ok_or(anyhow!("result data not found"))
    }
}
