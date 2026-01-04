use std::{
    f32::consts,
    fs::{self, File},
    io::{ErrorKind, Read, Write},
    time::Instant,
};

use indicatif::ProgressBar;
use rand::seq::SliceRandom;
use raylib::{
    color::Color,
    ffi::{KeyboardKey, MouseButton},
    prelude::RaylibDraw,
};

use crate::{
    network::{Network, NetworkBuilder},
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

fn print_network_stats(
    trainer: &mut Trainer,
    dataset: &Vec<DataPoint>,
    test_dataset: &Vec<DataPoint>,
) {
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

    let mut total_test_loss = 0.0;
    let mut total_test_correct = 0;
    for data_pt in test_dataset {
        let output = trainer.network().forward_inference(&data_pt.input);
        let loss = trainer.loss_fn().forward(&output, &data_pt.target);
        total_test_loss += loss;
        if max_index(&output) == max_index(&data_pt.target) {
            total_test_correct += 1;
        }
    }

    let loss = total_loss / dataset.len() as f32;
    let accuracy = total_correct as f32 / dataset.len() as f32;
    let test_loss = total_test_loss / test_dataset.len() as f32;
    let test_accuracy = total_test_correct as f32 / test_dataset.len() as f32;
    println!("Network loss: {}", loss);
    println!("Network accuracy: {}", accuracy);
    println!("Network test loss: {}", test_loss);
    println!("Network test accuracy: {}", test_accuracy);
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

fn load_network(mut file: File) -> Option<Network> {
    let mut buffer = Vec::new();
    match file.read_to_end(&mut buffer) {
        Ok(_) => {}
        Err(err) => {
            println!("Error reading file mnist-net.bin: {}", err.to_string());
            return None;
        }
    }
    let network = wincode::deserialize::<Network>(&buffer).expect("Could not deserialize network");
    println!("Loaded network file");

    Some(network)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut max = logits[0];
    for i in 1..logits.len() {
        max = max.max(logits[i]);
    }

    let mut exp_sum = 0.0;
    for logit in logits {
        exp_sum += (logit - max).exp();
    }

    let mut result = Vec::with_capacity(logits.len());
    for logit in logits {
        result.push((logit - max).exp() / exp_sum);
    }
    result
}

fn run_drawing_program(network: Network) {
    let (mut rl, thread) = raylib::init()
        .size(1350, 700)
        .title("Hello World")
        .vsync()
        .build();

    const WIDTH: u32 = 28;
    const HEIGHT: u32 = 28;
    const DRAW_RADIUS: f32 = 2.0;
    let mut drawing_buffer = vec![0.0f32; (WIDTH * HEIGHT) as usize];

    while !rl.window_should_close() {
        if rl.is_key_pressed(KeyboardKey::KEY_C) {
            drawing_buffer.fill(0.0);
        }
        if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT) {
            let mouse_x = (rl.get_mouse_x() as f32) / 25.0;
            let mouse_y = (rl.get_mouse_y() as f32) / 25.0;
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let pixel = &mut drawing_buffer[(y * WIDTH + x) as usize];
                    let dist =
                        (mouse_x - (x as f32 + 0.5)).powi(2) + (mouse_y - (y as f32 + 0.5)).powi(2);
                    if dist <= DRAW_RADIUS {
                        *pixel = 1.0;
                    } else if dist <= DRAW_RADIUS + 1.5 {
                        *pixel = pixel.max((dist - DRAW_RADIUS) / 1.5);
                    }
                }
            }
        }

        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::WHITE);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                // clamp probably not necessary but I'm being safe
                let brightness =
                    (drawing_buffer[(y * WIDTH + x) as usize] * 255.0).clamp(0.0, 255.0) as u8;
                d.draw_rectangle(
                    x as i32 * 25,
                    y as i32 * 25,
                    25,
                    25,
                    Color::new(brightness, brightness, brightness, 255),
                )
            }
        }

        let logits = network.forward_inference(&drawing_buffer);
        let result = softmax(&logits);
        let mut indices: Vec<usize> = (0..10).collect();
        indices.sort_by(|&a, &b| result[b].partial_cmp(&result[a]).unwrap());

        for (idx, &digit) in indices.iter().enumerate() {
            d.draw_rectangle(950, 70 * idx as i32 + 10, 300, 50, Color::DARKGRAY);
            d.draw_rectangle(
                960,
                70 * idx as i32 + 20,
                (280.0 * result[digit as usize]) as i32,
                30,
                Color::WHITE,
            );
            let percent = format!("{:.2}%", result[digit as usize] * 100.0);
            d.draw_text(&percent, 710, 70 * idx as i32 + 3, 70, Color::BLACK);

            let digit_char = (digit as u8 + '0' as u8) as char;
            let mut buf = [0u8; 4];
            d.draw_text(
                digit_char.encode_utf8(&mut buf),
                1280,
                70 * idx as i32 + 3,
                70,
                Color::BLACK,
            );
        }
    }
}

fn augment_image(pixels: &[f32]) -> Vec<f32> {
    let mut new_image = vec![0.0f32; 784];

    let translation_x = rand::random_range(-4.0f32..=4.0f32);
    let translation_y = rand::random_range(-4.0f32..=4.0f32);

    let rotation = rand::random_range(-12.5f32..=12.5f32) * consts::PI / 180.0;

    let scale = rand::random_range(0.85f32..=1.15f32);

    for y in 0..28 {
        for x in 0..28 {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            let mut transformed_x = (px - 14.0) * rotation.cos() + (py - 14.0) * rotation.sin();
            let mut transformed_y = (py - 14.0) * rotation.cos() - (px - 14.0) * rotation.sin();

            transformed_x *= scale;
            transformed_y *= scale;

            transformed_x += translation_x + 14.0;
            transformed_y += translation_y + 14.0;

            transformed_x = transformed_x.clamp(0.5, 27.5);
            transformed_y = transformed_y.clamp(0.5, 27.5);

            let sample_x = (transformed_x - 0.5).floor() as usize;
            let sample_y = (transformed_y - 0.5).floor() as usize;
            let factor_x = (transformed_x - 0.5).fract();
            let factor_y = (transformed_y - 0.5).fract();

            let ll = pixels[sample_y * 28 + sample_x];
            let ul = pixels[sample_y * 28 + (sample_x + 1).min(27)];
            let lu = pixels[(sample_y + 1).min(27) * 28 + sample_x];
            let uu = pixels[(sample_y + 1).min(27) * 28 + (sample_x + 1).min(27)];
            let ly = ll * (1.0 - factor_x) + ul * factor_x;
            let uy = lu * (1.0 - factor_x) + uu * factor_x;
            let sample = ly * (1.0 - factor_y) + uy * factor_y;
            new_image[y * 28 + x] = sample;
        }
    }

    for _ in 0..8 {
        new_image[rand::random_range(0..784) as usize] = rand::random_range(0.0..=1.0);
    }

    new_image
}

fn visualize_augmentations(dataset: &Vec<DataPoint>) {
    let (mut rl, thread) = raylib::init()
        .size(1500, 700)
        .title("Hello World")
        .vsync()
        .build();

    const WIDTH: u32 = 28;
    const HEIGHT: u32 = 28;
    let mut curr_idx = 0usize;
    let mut curr_image = &dataset[curr_idx].input;
    let mut curr_augmented_image = augment_image(&dataset[curr_idx].input);
    let mut curr_label = max_index(&dataset[curr_idx].target);

    while !rl.window_should_close() {
        if rl.is_key_pressed(KeyboardKey::KEY_N) {
            curr_idx += 1;
            curr_image = &dataset[curr_idx].input;
            curr_augmented_image = augment_image(&dataset[curr_idx].input);
            curr_label = max_index(&dataset[curr_idx].target);
        }

        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::WHITE);
        let drawing_buffer = curr_image;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                // clamp probably not necessary but I'm being safe
                let brightness =
                    (drawing_buffer[(y * WIDTH + x) as usize] * 255.0).clamp(0.0, 255.0) as u8;
                d.draw_rectangle(
                    x as i32 * 25,
                    y as i32 * 25,
                    25,
                    25,
                    Color::new(brightness, brightness, brightness, 255),
                )
            }
        }

        let drawing_buffer = &curr_augmented_image;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                // clamp probably not necessary but I'm being safe
                let brightness =
                    (drawing_buffer[(y * WIDTH + x) as usize] * 255.0).clamp(0.0, 255.0) as u8;
                d.draw_rectangle(
                    x as i32 * 25 + 710,
                    y as i32 * 25,
                    25,
                    25,
                    Color::new(brightness, brightness, brightness, 255),
                )
            }
        }

        let digit_char = (curr_label as u8 + '0' as u8) as char;
        let mut buf = [0u8; 4];
        d.draw_text(
            digit_char.encode_utf8(&mut buf),
            1420,
            300,
            100,
            Color::BLACK,
        );
    }
}

fn main() {
    match File::open("mnist-net.bin") {
        Ok(file) => {
            println!("Loading network mnist-net.bin");
            if let Some(network) = load_network(file) {
                run_drawing_program(network);
            }
            return;
        }
        Err(err) => {
            if err.kind() == ErrorKind::NotFound {
                println!("No network found");
            } else {
                println!("Could not open file mnist-net.bin: {}", err.to_string());
                return;
            }
        }
    }

    let args: Vec<String> = std::env::args().collect();
    let mnist_dir = args[1].clone();
    let mut dataset = load_mnist_dataset(
        (mnist_dir.clone() + "/train-images-idx3-ubyte").as_str(),
        (mnist_dir.clone() + "/train-labels-idx1-ubyte").as_str(),
    )
    .expect("Could not load MNIST dataset");
    let test_dataset = load_mnist_dataset(
        (mnist_dir.clone() + "/t10k-images-idx3-ubyte").as_str(),
        (mnist_dir.clone() + "/t10k-labels-idx1-ubyte").as_str(),
    )
    .expect("Could not load MNIST test dataset");

    if args.len() > 2 && args[2] == "visualize" {
        println!("Visualizing dataset augmentations");
        visualize_augmentations(&dataset);
        return;
    }

    println!("Training a network");

    let mut network = NetworkBuilder::new(784)
        .add_dense_layer(256)
        .add_relu()
        .add_dense_layer(64)
        .add_relu()
        .add_dense_layer(10)
        .build();
    network.init_rand();

    const BATCH_SIZE: u32 = 32;

    // let mut trainer = Trainer::new(network, BATCH_SIZE);
    let mut trainer = TrainerBuilder::new(network)
        .adamw(0.001, 0.003)
        .batch_size(BATCH_SIZE)
        .cross_entropy()
        .build();

    print_network_stats(&mut trainer, &dataset, &test_dataset);

    dataset.shuffle(&mut rand::rng());

    for i in 0..70 {
        let num_batches = dataset.len() as u32 / BATCH_SIZE;
        let start_time = Instant::now();

        let bar = ProgressBar::new(num_batches as u64);
        for i in 0..dataset.len() as u32 / BATCH_SIZE {
            let begin = (i * BATCH_SIZE) as usize;
            let end = ((i + 1) * BATCH_SIZE) as usize;
            let batch = &dataset[begin..end];
            trainer.run_batch_augmented(batch, augment_image);
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
        print_network_stats(&mut trainer, &dataset, &test_dataset);
    }

    let data = wincode::serialize(trainer.network()).expect("Could not serialize network");
    let mut net_file =
        File::create("mnist-net.bin").expect("Could not open file mnist-net.bin for writing");
    match net_file.write_all(&data) {
        Ok(()) => {
            println!("Wrote network to file mnist-net.bin");
        }
        Err(err) => {
            println!("Error opening file: {}", err.to_string());
        }
    }
}
