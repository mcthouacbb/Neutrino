use crate::tensor::{Shape, Tensor};

pub struct ReluLayer {
    size: u32,
}

impl ReluLayer {
    pub fn new(size: u32) -> Self {
        Self { size }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn forward(&self, inputs: &Tensor) -> Tensor {
        assert!(*inputs.shape() == Shape::vector(self.size()));

        let mut result = inputs.clone();
        for i in 0..result.shape().dim(0) {
            result[i] = result[i].max(0.0);
        }
        result
    }

    pub fn backward(&self, output_grads: &Tensor, inputs: &Tensor) -> Tensor {
        let mut input_grads = output_grads.clone();
        for i in 0..self.size {
            if inputs[i] < 0.0 {
                input_grads[i] = 0.0;
            }
        }
        input_grads
    }
}
