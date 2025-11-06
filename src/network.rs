use rand::Rng;

use crate::{layer::DenseLayer, vector::Vector};

pub struct Network {
    layers: Vec<DenseLayer>
}

impl Network {
    pub fn new<const N: usize>(layer_sizes: [u32; N]) -> Network {
        assert!(N > 1);
        let mut result = Self {
            layers: Vec::new()
        };
        for i in 0..(N - 1) {
            result.layers.push(DenseLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        result
    }

    pub fn forward(&self, inputs: &Vector) -> Vector {
        let mut output = self.layers[0].forward(inputs);
        for layer in self.layers.iter().skip(1) {
            output = layer.forward(&output);
        }
        output
    }

    pub fn init_rand(&mut self) {
        for layer in &mut self.layers {
            for weight in layer.weights_mut().elems_mut() {
                *weight = rand::rng().random_range(-1.0..=1.0);
            }
        }
    }
}
