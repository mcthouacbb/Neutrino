mod dense_layer;
pub use dense_layer::*;

use crate::vector::Vector;

pub trait Layer {
    fn init_rand(&mut self);
    fn forward(&self, inputs: &Vector) -> Vector;
}
