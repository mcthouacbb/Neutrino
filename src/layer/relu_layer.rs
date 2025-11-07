use super::Layer;
use crate::vector::Vector;

pub struct ReluLayer {
    size: u32,
}

impl ReluLayer {
    pub fn new(size: u32) -> Self {
        Self { size: size }
    }
}

impl Layer for ReluLayer {
    fn input_size(&self) -> u32 {
        self.size
    }

    fn output_size(&self) -> u32 {
        self.size
    }

    fn init_rand(&mut self) {}

    fn forward(&self, inputs: &Vector) -> Vector {
        assert!(inputs.len() == self.input_size());

        let mut result = inputs.clone();
        for i in 0..result.len() {
            result[i] = result[i].max(0.0);
        }
        result
    }
}
