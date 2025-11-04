use crate::{network::Network, vector::Vector};

mod network;
mod layer;
mod matrix;
mod vector;

fn main() {
    println!("Hello, world!");

    let network = Network::new([3, 6, 5, 1]);
    let mut inputs = Vector::zeros(3);
    inputs[0] = 5.0;
    inputs[1] = 3.0;
    inputs[2] = 1.0;
    let output = network.forward(&inputs)[0];
    println!("Network output: {}", output);
}
