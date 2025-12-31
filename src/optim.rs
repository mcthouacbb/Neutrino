mod sgd;

pub use sgd::*;

use std::ops::Range;

use crate::tensor::Tensor;

pub trait Optimizer {
    fn update_range(
        &mut self,
        backwardables: &mut [Tensor],
        grads: &[Tensor],
        idx_range: Range<usize>,
        batch_size: u32,
    );
}
