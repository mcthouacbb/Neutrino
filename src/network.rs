use crate::{layer::Layer, vector::Vector};

pub struct Network {
    layers: Vec<Box<dyn Layer>>
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self {
            layers: layers
        }
    }
    
    pub fn forward(&self, inputs: &Vector) -> Vector {
        let mut output = self.layers[0].forward(inputs);
        for layer in self.layers.iter().skip(1) {
            output = layer.forward(&output);
        }
        output
    }

    pub fn init_rand(&mut self) {
        for layer in &mut self.layers {
            layer.init_rand();
        }
    }
}
