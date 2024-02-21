<a name="readme-top"></a>

# Plugin examples for Jobworkerp-rs runner with candle

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About
  This is plugin examples of [jobworkerp-rs](https://github.com/jobworkerp-rs/jobworkerp-rs/) with transformer implement: [Candle](https://github.com/huggingface/candle).

  run transformer models from huggingface repository or local file:
  - SentenceBertRunner (Bert): calculate sentence embedding vector with Bert
    - executable binary file: embedding-test
  - LLMPromptRunner (LLaMA, StableLM, Mistral, Falcon): LLM inference runner
    - executable binary file: llm-test
  - WhisperRunner (whisper): Transcribe runner (mp3 format supported)
    - executable binary file: whisper-test

note: Only models supported by the candle library can be successfully executed.

## Getting Started


### Prerequisites

- build: cuda and cudnn library installed
- enable cuda for container if use [container image](https://github.com/jobworkerp-rs/jobworkerp-rs/pkgs/container/jobworkerp)
  - [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- (using docker image) Run on linux x86_64 machine

### Installation

_build binary_

- Build with cargo:
  ```
  cargo build --release
  ```

_use container image_

- Pull image from github:
  ```
  docker pull ghcr.io/jobworkerp-rs/jobworkerp:latest
  ```


## Usage Examples

- use as plugins of jobworkerp-rs: built by cargo or download plugin binary files to directory: PLUGINS_RUNNER_DIR (`./plugins` in this instruction)
- Example for running with hybrid storage(redis and rdb): use docker-compose.yaml (using redis and sqlite3). example: 
  ```
   $ mkdir log  # to store log file
   $ mkdir cache  # huggingface cache directory
   $ chmod 777 log cache
   $ docker-compose up
   (take a few minutes for downloading and loading model files from huggingface at the first boot. do next step after confirming log message printed:  "load Runner plugin file:" for all plugins and after printing message: "create job dispatcher for channel __default_job_channel__", runners is ready for request)
  ```
  - The example of running llm plugin runner as worker with docker-compose.
    - register worker (worker for get DIRECT response)
    ```
     $ grpcurl -d '{"name":"LLMWorker","type":"PLUGIN","operation":"LLMPromptRunner","response_type":"DIRECT","next_workers":[],"retry_policy":{"type":"EXPONENTIAL","interval":"1000","max_interval":"60000","max_retry":"3","basis":"2"},"store_success":true,"store_failure":true}' \
     -plaintext localhost:9000 jobworkerp.service.WorkerService/Create
    ```
    - confirm registered worker
    ```
    $ grpcurl -d '{}' -plaintext localhost:9000 jobworkerp.service.WorkerService/FindList
    ```
    - enqueue job: execute runner (`arg` in request: base64 encoding of [protobuf/llm.proto](./protobuf/llm.proto): You will get base64 text with command: `./llm-test --prompt <propt> --request-output-only`)
    ```
    $ grpcurl -d '{"arg":"Cg9Zb3UgYXJlIHJpZ2h0ISAQ6ActmpmZPzBAQAE=","worker_name":"LLMWorker","timeout":"120000"}' \
    -plaintext localhost:9000 jobworkerp.service.JobService/Enqueue
    ```
    - will get result base64: result.data.output.items[0]

- Example for use executables samples
  - create .env: `cp dot.env .env`, and edit .env file
  - run commands:
  ```
  $ ./target/debug/llm-test --prompt <your prompt>
  $ ./target/debug/embedding-test --sentences <your sentences>
  $ ./target/debug/whisper-test --input <input file path or url>
  ```
  - get command detail: Run with --help option


## License

Distributed under the MIT License. See `LICENSE` for more information.
