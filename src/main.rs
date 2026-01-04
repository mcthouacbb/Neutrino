use std::{fs, time::Instant};

use indicatif::ProgressBar;
use rand::seq::SliceRandom;

use crate::{
    network::NetworkBuilder,
    trainer::{Trainer, TrainerBuilder},
};

mod layer;
mod loss;
mod network;
mod optim;
mod trainer;

#[derive(Clone)]
struct DataPoint {
    input: Vec<f32>,
    target: Vec<f32>,
}

fn max_index(t: &[f32]) -> usize {
    let mut max_val = -1.0;
    let mut max_idx = 0;
    for (idx, val) in t.iter().enumerate() {
        if *val > max_val {
            max_idx = idx;
            max_val = *val;
        }
    }
    max_idx
}

fn print_network_stats(trainer: &mut Trainer, dataset: &Vec<DataPoint>) {
    let mut total_loss = 0.0;
    let mut total_correct = 0;
    for data_pt in dataset {
        let output = trainer.network().forward_inference(&data_pt.input);
        let loss = trainer.loss_fn().forward(&output, &data_pt.target);
        total_loss += loss;
        if max_index(&output) == max_index(&data_pt.target) {
            total_correct += 1;
        }
    }
    let loss = total_loss / dataset.len() as f32;
    let accuracy = total_correct as f32 / dataset.len() as f32;
    println!("Network loss: {}", loss);
    println!("Network accuracy: {}", accuracy);
}

fn load_mnist_dataset(image_file: &str, label_file: &str) -> Option<Vec<DataPoint>> {
    let image_data = fs::read(image_file).ok()?;
    let label_data = fs::read(label_file).ok()?;

    let magic_num = u32::from_be_bytes(image_data[0..4].try_into().unwrap());
    if magic_num != 2051 {
        println!(
            "Magic number for image file {} does not match 2051",
            image_file
        );
        return None;
    }

    let num_images = u32::from_be_bytes(image_data[4..8].try_into().unwrap());
    let image_width = u32::from_be_bytes(image_data[8..12].try_into().unwrap());
    let image_height = u32::from_be_bytes(image_data[12..16].try_into().unwrap());

    if image_data.len() as u32 != 16 + num_images * image_width * image_height {
        println!(
            "Number of bytes does not match expected file size for image file {}",
            image_file
        );
        return None;
    }

    let magic_num = u32::from_be_bytes(label_data[0..4].try_into().unwrap());
    if magic_num != 2049 {
        println!(
            "Magic number for label file {} does not match 2049",
            label_file
        );
        return None;
    }

    let num_labels = u32::from_be_bytes(label_data[4..8].try_into().unwrap());
    if label_data.len() as u32 != 8 + num_labels {
        println!(
            "Number of bytes does not match expected file size for label file {}",
            label_file
        );
        return None;
    }

    if num_labels != num_images {
        println!(
            "Number of labels and number of images do not match for image file {} and label file {}",
            image_file, label_file
        );
        return None;
    }

    let mut result = Vec::new();
    for i in 0..num_images {
        let mut input = vec![0.0; (image_width * image_height) as usize];
        let image_data_base_idx = 16 + i * image_width * image_height;
        for y in 0..image_height {
            for x in 0..image_width {
                let offset = y * image_width + x;
                let image_data_idx = image_data_base_idx + offset;
                input[offset as usize] = image_data[image_data_idx as usize] as f32 / 255.0;
            }
        }

        let mut target = vec![0.0; 10];
        let label_data_idx = 8 + i;
        // one-hot encoding
        target[label_data[label_data_idx as usize] as usize] = 1.0;

        result.push(DataPoint { input, target })
    }
    Some(result)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mnist_dir = args[1].clone();
    let mut dataset = load_mnist_dataset(
        (mnist_dir.clone() + "/train-images-idx3-ubyte").as_str(),
        (mnist_dir + "/train-labels-idx1-ubyte").as_str(),
    )
    .expect("Could not load MNIST dataset");

    let mut network = NetworkBuilder::new(784)
        .add_dense_layer(256)
        .add_relu()
        .add_dense_layer(10)
        .build();
    network.init_rand();

    const BATCH_SIZE: u32 = 16;

    // let mut trainer = Trainer::new(network, BATCH_SIZE);
    let mut trainer = TrainerBuilder::new(network)
        .adamw(0.01, 0.003)
        .batch_size(BATCH_SIZE)
        .cross_entropy()
        .build();

    print_network_stats(&mut trainer, &dataset);

    dataset.shuffle(&mut rand::rng());

    for i in 0..100000 {
        let num_batches = dataset.len() as u32 / BATCH_SIZE;
        let start_time = Instant::now();

        let bar = ProgressBar::new(num_batches as u64);
        for i in 0..dataset.len() as u32 / BATCH_SIZE {
            let begin = (i * BATCH_SIZE) as usize;
            let end = ((i + 1) * BATCH_SIZE) as usize;
            let batch = &dataset[begin..end];
            trainer.run_batch(batch);
            bar.inc(1);
        }

        let end_time = Instant::now();

        bar.finish();

        let seconds = (end_time - start_time).as_secs_f64();
        println!("Epoch: {}", i);
        println!("time: {}", seconds);
        println!(
            "batches/s: {}, samples/s: {}",
            num_batches as f64 / seconds,
            (num_batches * BATCH_SIZE) as f64 / seconds
        );
        print_network_stats(&mut trainer, &dataset);
    }
}
