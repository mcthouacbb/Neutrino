use crate::{network::Network, optim::Optimizer};

pub struct AdamW {
    lr: f32,
    lambda: f32,
    momentum: Vec<f32>,
    velocities: Vec<f32>,
}

impl AdamW {
    pub fn new(lr: f32, lambda: f32, network: &Network) -> Self {
        Self {
            lr,
            lambda,
            momentum: vec![0.0; network.num_params() as usize],
            velocities: vec![0.0; network.num_params() as usize],
        }
    }
}

impl Optimizer for AdamW {
    fn update(&mut self, params: &mut [f32], grads: &[f32], batch_size: u32) {
        const BETA1: f32 = 0.9;
        const BETA2: f32 = 0.999;
        const EPSILON: f32 = 0.00000001;

        let lr = self.lr / batch_size as f32;
        for (idx, param) in params.iter_mut().enumerate() {
            let velocity = &mut self.velocities[idx];
            let momentum = &mut self.momentum[idx];
            let gradient = &grads[idx];

            *momentum = BETA1 * *momentum + (1.0 - BETA1) * gradient;
            *velocity = BETA2 * *velocity + (1.0 - BETA2) * gradient * gradient;

            // TODO: bias-corrected momentum and velocity estimates
            *param -= *momentum / (velocity.sqrt() + EPSILON) * lr + *param * self.lambda * lr;
        }
    }
}
