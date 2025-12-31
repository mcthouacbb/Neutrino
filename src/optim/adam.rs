use std::ops::Range;

use crate::{network::Network, optim::Optimizer, tensor::Tensor};

pub struct Adam {
    lr: f32,
    momentum: Vec<Tensor>,
    velocities: Vec<Tensor>,
}

impl Adam {
    pub fn new(lr: f32, network: &Network) -> Self {
        Self {
            lr,
            momentum: network.zero_grads().0,
            velocities: network.zero_grads().0,
        }
    }
}

impl Optimizer for Adam {
    fn update_range(
        &mut self,
        backwardables: &mut [Tensor],
        grads: &[Tensor],
        idx_range: Range<usize>,
        batch_size: u32,
    ) {
        const BETA1: f32 = 0.9;
        const BETA2: f32 = 0.999;
        const EPSILON: f32 = 0.00000001;

        let lr = self.lr / batch_size as f32;
        for (slice_idx, adam_idx) in idx_range.enumerate() {
            let backwardable = &mut backwardables[slice_idx];
            let grad = &grads[slice_idx];
            let velocity = &mut self.velocities[adam_idx];
            let momentum = &mut self.momentum[adam_idx];
            for (idx, elem) in backwardable.elems_mut().iter_mut().enumerate() {
                let velocity_elem = &mut velocity.elems_mut()[idx];
                let momentum_elem = &mut momentum.elems_mut()[idx];
                let grad_elem = grad.elems()[idx];
                *momentum_elem = BETA1 * *momentum_elem + (1.0 - BETA1) * grad_elem;
                *velocity_elem = BETA2 * *velocity_elem + (1.0 - BETA2) * grad_elem * grad_elem;

                // TODO: bias-corrected momentum and velocity estimates
                *elem -= *momentum_elem / (velocity_elem.sqrt() + EPSILON) * lr;
            }
        }
    }
}
