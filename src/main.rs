use crate::{
    layer::DenseLayer,
    network::{Network, NetworkBuilder},
    tensor::{Shape, Tensor},
};

mod layer;
mod network;
mod tensor;

fn main() {
    println!("Hello, world!");

    let mut builder = NetworkBuilder::new(3);
    builder.add_dense_layer(1);
    builder.add_relu();
    let mut network = builder.build();
    network.init_rand();
    let mut inputs = Tensor::zeros(Shape::vector(3));
    inputs[0] = 5.0;
    inputs[1] = 3.0;
    inputs[2] = 1.0;
    let output = network.forward(&inputs)[0];
    println!("Network output: {}", output);
}
