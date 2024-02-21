extern crate prost_build;
fn main() {
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["embedding.proto"], &["../protobuf"])
        .unwrap();
}
