use crate::{
    layer::{DenseLayer, Layer, ReluLayer},
    loss::{Loss, Mse},
    tensor::{Shape, Tensor},
};

pub struct NetworkGrads(Vec<Tensor>);

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    loss_fn: Box<dyn Loss>,
    num_backwardables: u32,
}

impl Network {
    fn new(layers: Vec<Box<dyn Layer>>, num_backwardables: u32) -> Self {
        let mut result = Self {
            layers,
            loss_fn: Box::new(Mse::new()),
            num_backwardables,
        };
        result
    }

    pub fn forward_all(&self, inputs: &Tensor) -> Vec<Tensor> {
        let mut result = vec![inputs.clone()];
        for layer in self.layers.iter() {
            result.push(layer.forward(result.last().unwrap()));
        }
        result
    }

    pub fn forward_all_loss(&self, inputs: &Tensor, targets: &Tensor) -> (Vec<Tensor>, f32) {
        let result = self.forward_all(inputs);
        let loss = self.loss_fn.forward(result.last().unwrap(), targets);
        (result, loss)
    }

    pub fn forward_loss(&self, inputs: &Tensor, targets: &Tensor) -> f32 {
        self.forward_all_loss(inputs, targets).1
    }

    pub fn forward(&self, inputs: &Tensor) -> Tensor {
        // clone probably not optimal here
        // but idk how to fix it
        self.forward_all(inputs).last().unwrap().clone()
    }

    pub fn backward(&mut self, inputs: &Tensor, targets: &Tensor, grads: &mut NetworkGrads) {
        let (outputs, loss) = self.forward_all_loss(inputs, targets);

        let mut output_grads = self.loss_fn.backward(outputs.last().unwrap(), targets);
        for (idx, layer) in self.layers.iter().enumerate().rev() {
            let layer_grads = &mut grads.0[layer.grad_idx_range()];
            output_grads = layer.backward(&output_grads, &outputs[idx], layer_grads);
        }
    }

    pub fn update(&mut self, grads: &NetworkGrads, lr: f32) {
        for layer in self.layers.iter_mut() {
            layer.update(&grads.0[layer.grad_idx_range()], lr);
        }
    }

    pub fn init_rand(&mut self) {
        for layer in &mut self.layers {
            layer.init_rand();
        }
    }

    pub fn zero_grads(&self) -> NetworkGrads {
        let mut result = NetworkGrads(vec![
            Tensor::zeros(Shape::scalar());
            self.num_backwardables as usize
        ]);
        for layer in &self.layers {
            layer.zero_grads(&mut result.0[layer.grad_idx_range()]);
        }
        result
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
        Network::new(self.layers, self.num_backwardables)
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
