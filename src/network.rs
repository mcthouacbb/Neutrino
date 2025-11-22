use crate::{
    layer::{DenseLayer, Layer, ReluLayer},
    tensor::Tensor,
};

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    pub fn forward(&self, inputs: &Tensor) -> Tensor {
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

pub struct NetworkBuilder {
    input_size: u32,
    num_backwardables: u32,
    layers: Vec<Box<dyn Layer>>,
}

impl NetworkBuilder {
    pub fn new(input_size: u32) -> Self {
        Self {
            input_size,
            num_backwardables: 0,
            layers: Vec::new(),
        }
    }

    pub fn add_dense_layer(&mut self, output_size: u32) {
        self.add_layer(Box::new(DenseLayer::new(
            self.next_input_size(),
            output_size,
            self.num_backwardables,
        )));
    }

    pub fn add_relu(&mut self) {
        self.add_layer(Box::new(ReluLayer::new(self.next_input_size())));
    }

    pub fn build(self) -> Network {
        Network::new(self.layers)
    }

    fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.num_backwardables += layer.num_backwardables();
        self.layers.push(layer);
    }

    fn next_input_size(&self) -> u32 {
        if let Some(last_layer) = self.layers.last() {
            last_layer.as_ref().output_size()
        } else {
            self.input_size
        }
    }
}
