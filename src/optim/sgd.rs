use crate::optim::Optimizer;

pub struct Sgd {
    lr: f32,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl Optimizer for Sgd {
    fn update(&mut self, params: &mut [f32], grads: &[f32], batch_size: u32) {
        let lr = self.lr / batch_size as f32;
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            *param -= grad * lr;
        }
    }
}
