use crate::{loss::Loss, tensor::Tensor};

pub struct Mse {}

impl Mse {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for Mse {
    fn forward(&self, inputs: &Tensor, targets: &Tensor) -> f32 {
        assert!(*inputs.shape() == *targets.shape());

        let mut result = 0.0;
        for (input, target) in inputs.elems().iter().zip(targets.elems().iter()) {
            let diff = *input - *target;
            result += diff * diff;
        }
        result
    }

    fn backward(&self, inputs: &Tensor, targets: &Tensor) -> Tensor {
        assert!(*inputs.shape() == *targets.shape());

        let mut result = Tensor::zeros(*targets.shape());
        for (i, v) in result.elems_mut().iter_mut().enumerate() {
            let diff = inputs.elems()[i] - targets.elems()[i];
            *v = 2.0 * diff;
        }
        result
    }
}
