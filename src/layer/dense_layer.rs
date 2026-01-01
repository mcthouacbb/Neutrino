use rand::Rng;

use super::Layer;
use crate::tensor::{Shape, Tensor};

pub struct DenseLayer {
    input_size: u32,
    output_size: u32,
    backwardables: [Tensor; 2],
    backwardable_idx: u32,
}

impl DenseLayer {
    pub fn new(input_size: u32, output_size: u32, backwardable_idx: u32) -> Self {
        Self {
            input_size,
            output_size,
            backwardables: [
                // weights
                Tensor::zeros(Shape::matrix(output_size, input_size)),
                // biases
                Tensor::zeros(Shape::vector(output_size)),
            ],
            backwardable_idx: backwardable_idx,
        }
    }

    pub fn weights(&self) -> &Tensor {
        &self.backwardables[0]
    }

    pub fn weights_mut(&mut self) -> &mut Tensor {
        &mut self.backwardables[0]
    }

    pub fn biases(&self) -> &Tensor {
        &self.backwardables[1]
    }

    pub fn biases_mut(&mut self) -> &mut Tensor {
        &mut self.backwardables[1]
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
        let bound = (6.0 / self.input_size as f32).sqrt();
        for weight in self.weights_mut().elems_mut() {
            *weight = rand::rng().random_range(-bound..=bound);
        }
    }

    fn num_backwardables(&self) -> u32 {
        2
    }

    fn backwardable_idx(&self) -> u32 {
        self.backwardable_idx
    }

    fn backwardables(&self) -> &[Tensor] {
        &self.backwardables
    }

    fn backwardables_mut(&mut self) -> &mut [Tensor] {
        &mut self.backwardables
    }

    fn zero_grads(&self, grads: &mut [Tensor]) {
        assert!(grads.len() as u32 == self.num_backwardables());
        grads[0] = Tensor::zeros(*self.weights().shape());
        grads[1] = Tensor::zeros(*self.biases().shape());
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        assert!(*inputs.shape() == Shape::vector(self.input_size()));

        let mut result = self.biases().clone();
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                result[i] += inputs[j] * self.weights()[(i, j)];
            }
        }
        result
    }

    fn backward(
        &self,
        output_grads: &Tensor,
        inputs: &Tensor,
        result_grads: &mut [Tensor],
    ) -> Tensor {
        // bias gradients
        for i in 0..self.output_size {
            result_grads[1][i] += output_grads[i];
        }

        // weight gradients
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                result_grads[0][(i, j)] += inputs[j] * output_grads[i];
            }
        }

        let mut input_grads = Tensor::zeros(Shape::vector(self.input_size));
        for j in 0..self.input_size {
            for i in 0..self.output_size {
                input_grads[j] += output_grads[i] * self.weights()[(i, j)];
            }
        }
        input_grads
    }

    fn update(&mut self, grads: &[Tensor], lr: f32) {
        for i in 0..self.output_size * self.input_size {
            self.weights_mut().elems_mut()[i as usize] -= lr * grads[0].elems()[i as usize];
        }

        for i in 0..self.output_size {
            self.biases_mut().elems_mut()[i as usize] -= lr * grads[1].elems()[i as usize];
        }
    }
}
