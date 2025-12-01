mod mse;

pub use mse::Mse;

use crate::tensor::Tensor;

pub trait Loss {
    fn forward(&self, inputs: &Tensor, targets: &Tensor) -> f32;
    fn backward(&self, inputs: &Tensor, targets: &Tensor) -> Tensor;
}

enum LossFn {
    Mse,
}
