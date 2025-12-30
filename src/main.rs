use crate::{
    layer::DenseLayer,
    network::{Network, NetworkBuilder},
    tensor::{Shape, Tensor},
};

mod layer;
mod loss;
mod network;
mod tensor;

fn main() {
    println!("Hello, world!");

    let mut builder = NetworkBuilder::new(3);
    builder.add_dense_layer(5);
    builder.add_relu();
    builder.add_dense_layer(4);
    builder.add_relu();
    builder.add_dense_layer(1);
    let mut network = builder.build();
    network.init_rand();
    let mut inputs = Tensor::zeros(Shape::vector(3));
    inputs[0] = 5.0;
    inputs[1] = 3.0;
    inputs[2] = 1.0;
    let target = Tensor::zeros(Shape::vector(1));
    let output = network.forward(&inputs)[0];
    let loss = network.forward_loss(&inputs, &target);
    println!("Network output: {}", output);
    println!("Network loss: {}", loss);

    for i in 0..100 {
        let mut grads = network.zero_grads();
        network.backward(&inputs, &target, &mut grads);
        network.update(&grads, 0.01);

        let new_output = network.forward(&inputs)[0];
        let new_loss = network.forward_loss(&inputs, &target);

        println!("Network output: {}", new_output);
        println!("Network loss: {}", new_loss);
    }
}
