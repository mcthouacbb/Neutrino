use crate::optim::Optimizer;

pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr: lr }
    }
}

impl Optimizer for SGD {
    fn update_range(
        &mut self,
        backwardables: &mut [crate::tensor::Tensor],
        grads: &[crate::tensor::Tensor],
        idx_range: std::ops::Range<usize>,
        batch_size: u32,
    ) {
        let lr = self.lr / batch_size as f32;
        for (backwardable, grad) in backwardables.iter_mut().zip(grads.iter()) {
            for (elem, grad_elem) in backwardable.elems_mut().iter_mut().zip(grad.elems().iter()) {
                *elem -= grad_elem * lr;
            }
        }
    }
}
