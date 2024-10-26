extern crate prost_build;
fn main() {
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["protobuf/llm/llm_operation.proto","protobuf/llm/llm_arg.proto"], &["protobuf"])
        .unwrap();
}
