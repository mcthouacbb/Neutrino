use std::sync::atomic::AtomicU32;

use crate::{
    layer::{DenseLayer, Layer, ReluLayer},
    loss::{Loss, Mse},
    optim::Optimizer,
    tensor::{Shape, Tensor},
};

pub struct NetworkGrads(pub Vec<Tensor>);

pub struct Network {
    layers: Vec<Layer>,
    loss_fn: Box<dyn Loss>,
    num_backwardables: u32,
}

impl Network {
    fn new(layers: Vec<Layer>, num_backwardables: u32) -> Self {
        let result = Self {
            layers,
            loss_fn: Box::new(Mse::new()),
            num_backwardables,
        };
        result
    }

    pub fn forward_all(&self, inputs: &Tensor) -> Vec<Tensor> {
        let mut result = vec![inputs.clone()];
        for layer in self.layers.iter() {
            match layer {
                Layer::Dense(dense_layer) => {
                    result.push(dense_layer.forward(result.last().unwrap()));
                }
                Layer::ReLu(relu_layer) => {
                    result.push(relu_layer.forward(result.last().unwrap()));
                }
            }
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
            match layer {
                Layer::Dense(dense_layer) => {
                    let layer_grads: &mut [Tensor] = &mut grads.0[dense_layer.grad_idx_range()];
                    output_grads = dense_layer.backward(&output_grads, &outputs[idx], layer_grads);
                }
                Layer::ReLu(relu_layer) => {
                    output_grads = relu_layer.backward(&output_grads, &outputs[idx]);
                }
            }
        }
    }

    pub fn update(&mut self, grads: &NetworkGrads, optim: &mut dyn Optimizer, batch_size: u32) {
        for layer in self.layers.iter_mut() {
            match layer {
                Layer::Dense(dense_layer) => {
                    let grad_idx_range = dense_layer.grad_idx_range();
                    optim.update_range(
                        dense_layer.backwardables_mut(),
                        &grads.0[grad_idx_range.clone()],
                        grad_idx_range,
                        batch_size,
                    );
                }
                _ => {}
            }
        }
    }

    pub fn init_rand(&mut self) {
        for layer in &mut self.layers {
            match layer {
                Layer::Dense(dense_layer) => dense_layer.init_rand(),
                _ => {}
            }
        }
    }

    pub fn zero_grads(&self) -> NetworkGrads {
        let mut result = NetworkGrads(vec![
            Tensor::zeros(Shape::scalar());
            self.num_backwardables as usize
        ]);
        for layer in &self.layers {
            match layer {
                Layer::Dense(dense_layer) => {
                    dense_layer.zero_grads(&mut result.0[dense_layer.grad_idx_range()])
                }
                _ => {}
            }
        }
        result
    }
}

pub struct NetworkBuilder {
    input_size: u32,
    num_backwardables: u32,
    layers: Vec<Layer>,
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
        self.add_layer(Layer::Dense(DenseLayer::new(
            self.next_input_size(),
            output_size,
            self.num_backwardables,
        )));
    }

    pub fn add_relu(&mut self) {
        self.add_layer(Layer::ReLu(ReluLayer::new(self.next_input_size())));
    }

    pub fn build(self) -> Network {
        Network::new(self.layers, self.num_backwardables)
    }

    fn add_layer(&mut self, layer: Layer) {
        match &layer {
            Layer::Dense(dense_layer) => {
                self.num_backwardables += dense_layer.num_backwardables();
            }
            _ => {}
        }
        self.layers.push(layer);
    }

    fn next_input_size(&self) -> u32 {
        if let Some(last_layer) = self.layers.last() {
            match last_layer {
                Layer::Dense(dense_layer) => dense_layer.output_size(),
                Layer::ReLu(relu_layer) => relu_layer.size(),
            }
        } else {
            self.input_size
        }
    }
}
