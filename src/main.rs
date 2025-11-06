use crate::{layer::DenseLayer, network::Network, vector::Vector};

mod network;
mod layer;
mod matrix;
mod vector;

fn main() {
    println!("Hello, world!");

    let mut network = Network::new(vec![Box::new(DenseLayer::new(3, 1))]);
    network.init_rand();
    let mut inputs = Vector::zeros(3);
    inputs[0] = 5.0;
    inputs[1] = 3.0;
    inputs[2] = 1.0;
    let output = network.forward(&inputs)[0];
    println!("Network output: {}", output);
}
