use rand::Rng;

use super::Layer;
use crate::tensor::{Shape, Tensor};

pub struct DenseLayer {
    input_size: u32,
    output_size: u32,
    weights: Tensor,
    biases: Tensor,
    backwardable_idx: u32,
}

impl DenseLayer {
    pub fn new(input_size: u32, output_size: u32, backwardable_idx: u32) -> Self {
        Self {
            input_size,
            output_size,
            weights: Tensor::zeros(Shape::matrix(input_size, output_size)),
            biases: Tensor::zeros(Shape::vector(output_size)),
            backwardable_idx: backwardable_idx,
        }
    }

    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut Tensor {
        &mut self.weights
    }

    pub fn biases(&self) -> &Tensor {
        &self.biases
    }

    pub fn biases_mut(&mut self) -> &mut Tensor {
        &mut self.biases
    }
}

impl Layer for DenseLayer {
    fn input_size(&self) -> u32 {
        self.input_size
    }

    fn output_size(&self) -> u32 {
        self.output_size
    }

    fn init_rand(&mut self) {
        for weight in self.weights.elems_mut() {
            *weight = rand::rng().random_range(-1.0..=1.0);
        }
    }

    fn num_backwardables(&self) -> u32 {
        2
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        assert!(*inputs.shape() == Shape::vector(self.input_size()));

        let mut result = self.biases.clone();
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                result[i] += inputs[j] * self.weights[(i, j)];
            }
        }
        result
    }
}
