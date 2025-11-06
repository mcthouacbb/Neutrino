mod dense_layer;
pub use dense_layer::*;

use crate::vector::Vector;

pub trait Layer {
    fn input_size(&self) -> u32;
    fn output_size(&self) -> u32;

    fn init_rand(&mut self);
    fn forward(&self, inputs: &Vector) -> Vector;
}
