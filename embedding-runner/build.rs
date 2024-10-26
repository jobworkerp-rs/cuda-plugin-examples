extern crate prost_build;
fn main() {
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                "protobuf/embedding/embedding_operation.proto",
                "protobuf/embedding/embedding_arg.proto",
                "protobuf/embedding/embedding_result.proto",
            ],
            &["protobuf"],
        )
        .unwrap();
}
