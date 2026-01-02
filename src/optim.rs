mod adam;
mod adamw;
mod sgd;

pub use adam::*;
pub use adamw::*;
pub use sgd::*;

pub trait Optimizer {
    fn update(&mut self, params: &mut [f32], grads: &[f32], batch_size: u32);
}
