use std::{fs::File, io::Write};

use crate::{
    layer::DenseLayer,
    network::{Network, NetworkBuilder},
    tensor::{Shape, Tensor},
};

mod layer;
mod loss;
mod network;
mod tensor;

fn target_fn(mut x: f32) -> f32 {
    x = x / 10.0;
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        (256.0 * x.powi(5) - 640.0 * x.powi(4) + 512.0 * x.powi(3) - 128.0 * x.powi(2)) / 3.0 + x
    }
}

struct DataPoint {
    input: Tensor,
    target: Tensor,
}

fn get_data_points() -> Vec<DataPoint> {
    let mut result = Vec::new();
    for i in 0..100 {
        let x = i as f32 / 10.0;
        let mut input = Tensor::zeros(Shape::vector(1));
        input[0] = x;
        let mut target = Tensor::zeros(Shape::vector(1));
        target[0] = target_fn(x);
        result.push(DataPoint { input, target });
    }
    result
}

fn get_loss(network: &Network, data_points: &Vec<DataPoint>) -> f32 {
    let mut total_loss = 0.0;
    for data_pt in data_points {
        total_loss += network.forward_loss(&data_pt.input, &data_pt.target);
    }
    total_loss / data_points.len() as f32
}

fn main() {
    println!("Hello, world!");

    let mut builder = NetworkBuilder::new(1);
    builder.add_dense_layer(10);
    builder.add_relu();
    builder.add_dense_layer(8);
    builder.add_relu();
    builder.add_dense_layer(1);
    let mut network = builder.build();
    network.init_rand();
    let data_points = get_data_points();

    println!("Network loss: {}", get_loss(&network, &data_points));

    for i in 0..100000 {
        let mut grads = network.zero_grads();
        for data_pt in &data_points {
            network.backward(&data_pt.input, &data_pt.target, &mut grads);
        }
        network.update(&grads, 0.01 / data_points.len() as f32);

        if i % 100 == 0 {
            println!("Epoch: {}", i);
            println!("Network loss: {}", get_loss(&network, &data_points));
        }
    }

    for data_pt in &data_points {
        let result = network.forward(&data_pt.input);
        println!("Input: {}", data_pt.input[0]);
        println!("    Result: {}", result[0]);
        println!("    Target: {}", data_pt.target[0]);
    }

    // csv stuff
    let mut points_file = File::create("points.csv").unwrap();

    writeln!(points_file, "x,net,target");
    for data_pt in &data_points {
        let result = network.forward(&data_pt.input);
        writeln!(
            points_file,
            "{},{},{}",
            data_pt.input[0], result[0], data_pt.target[0]
        );
    }
}
