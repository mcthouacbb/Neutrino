use crate::{
    layer::Layer,
    loss::{Loss, Mse},
    network::Network,
    optim::{AdamW, Optimizer},
};

pub struct Trainer {
    network: Network,
    param_grad_buffer: Vec<f32>,
    value_buffer: Vec<Vec<f32>>,
    value_grad_buffer: Vec<Vec<f32>>,
    loss_fn: Box<dyn Loss>,
    optimizer: Box<dyn Optimizer>,
}

impl Trainer {
    pub fn new(network: Network) -> Self {
        let mut value_buffer = Vec::with_capacity(network.layers().len() + 1);
        value_buffer.push(vec![0.0; network.layers()[0].input_size() as usize]);
        for layer in network.layers() {
            value_buffer.push(vec![0.0; layer.output_size() as usize]);
        }

        let optimizer = AdamW::new(0.01, 0.003, &network);
        let num_params = network.num_params();
        Self {
            network: network,
            param_grad_buffer: vec![0.0; num_params as usize],
            value_buffer: value_buffer.clone(),
            value_grad_buffer: value_buffer,
            loss_fn: Box::new(Mse::new()),
            optimizer: Box::new(optimizer),
        }
    }

    pub fn network(&self) -> &Network {
        &self.network
    }

    pub fn loss_fn(&self) -> &dyn Loss {
        self.loss_fn.as_ref()
    }

    fn forward_all(&mut self, inputs: &[f32]) {
        self.value_buffer[0].copy_from_slice(inputs);

        for (idx, layer) in self.network.layers().iter().enumerate() {
            let (left, right) = self.value_buffer.split_at_mut(idx + 1);
            let inputs = &left[idx];
            let outputs = &mut right[0];
            match layer {
                Layer::Dense(dense_layer) => dense_layer.forward(
                    &self.network.param_buffer()[dense_layer.param_buffer_range()],
                    inputs,
                    outputs,
                ),
                Layer::ReLu(relu_layer) => {
                    relu_layer.forward(inputs, outputs);
                }
            }
        }
    }

    pub fn backward(&mut self, inputs: &[f32], targets: &[f32]) {
        self.forward_all(inputs);

        self.loss_fn.backward(
            self.value_buffer.last().unwrap(),
            targets,
            self.value_grad_buffer.last_mut().unwrap(),
        );
        for (idx, layer) in self.network.layers().iter().enumerate().rev() {
            let (left, right) = self.value_grad_buffer.split_at_mut(idx + 1);
            let output_grads = &right[0];
            let input_grads = &mut left[idx];
            match layer {
                Layer::Dense(dense_layer) => {
                    let layer_params =
                        &self.network.param_buffer()[dense_layer.param_buffer_range()];
                    let layer_grads = &mut self.param_grad_buffer[dense_layer.param_buffer_range()];

                    dense_layer.backward(
                        layer_params,
                        output_grads,
                        &self.value_buffer[idx],
                        layer_grads,
                        input_grads,
                    );
                }
                Layer::ReLu(relu_layer) => {
                    relu_layer.backward(output_grads, &self.value_buffer[idx], input_grads);
                }
            }
        }
    }

    pub fn update(&mut self, batch_size: u32) {
        self.optimizer.update(
            &mut self.network.param_buffer_mut(),
            &self.param_grad_buffer,
            batch_size,
        );
        self.param_grad_buffer.fill(0.0);
    }
}
