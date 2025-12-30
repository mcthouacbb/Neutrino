mod dense_layer;
mod relu_layer;
use std::ops::Range;

pub use dense_layer::*;
pub use relu_layer::*;

use crate::tensor::Tensor;

pub trait Layer {
    fn input_size(&self) -> u32;
    fn output_size(&self) -> u32;

    fn init_rand(&mut self);
    fn num_backwardables(&self) -> u32;
    fn backwardable_idx(&self) -> u32;
    fn grad_idx_range(&self) -> Range<usize> {
        (self.backwardable_idx() as usize)
            ..((self.backwardable_idx() + self.num_backwardables()) as usize)
    }
    fn zero_grads(&self, grads: &mut [Tensor]);
    fn forward(&self, inputs: &Tensor) -> Tensor;
    fn backward(
        &self,
        output_grads: &Tensor,
        inputs: &Tensor,
        result_grads: &mut [Tensor],
    ) -> Tensor;
}
