extern crate prost_build;
fn main() {
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                "protobuf/whisper/whisper_operation.proto",
                "protobuf/whisper/whisper_arg.proto",
                "protobuf/whisper/whisper_result.proto",
            ],
            &["protobuf"],
        )
        .unwrap();
}
