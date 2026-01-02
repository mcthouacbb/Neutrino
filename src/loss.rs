mod mse;

pub use mse::Mse;

pub trait Loss {
    fn forward(&self, inputs: &[f32], targets: &[f32]) -> f32;
    fn backward(&self, inputs: &[f32], targets: &[f32], grads: &mut [f32]);
}
