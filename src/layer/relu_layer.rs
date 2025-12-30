use super::Layer;
use crate::tensor::{Shape, Tensor};

pub struct ReluLayer {
    size: u32,
}

impl ReluLayer {
    pub fn new(size: u32) -> Self {
        Self { size }
    }
}

impl Layer for ReluLayer {
    fn input_size(&self) -> u32 {
        self.size
    }

    fn output_size(&self) -> u32 {
        self.size
    }

    fn init_rand(&mut self) {}

    fn num_backwardables(&self) -> u32 {
        0
    }

    fn backwardable_idx(&self) -> u32 {
        0
    }

    fn zero_grads(&self, grads: &mut [Tensor]) {
        assert!(grads.len() as u32 == self.num_backwardables());
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        assert!(*inputs.shape() == Shape::vector(self.input_size()));

        let mut result = inputs.clone();
        for i in 0..result.shape().dim(0) {
            result[i] = result[i].max(0.0);
        }
        result
    }

    fn backward(
        &self,
        output_grads: &Tensor,
        inputs: &Tensor,
        result_grads: &mut [Tensor],
    ) -> Tensor {
        let mut input_grads = output_grads.clone();
        for i in 0..self.size {
            if inputs[i] < 0.0 {
                input_grads[i] = 0.0;
            }
        }
        input_grads
    }

    fn update(&mut self, grads: &[Tensor], lr: f32) {}
}
