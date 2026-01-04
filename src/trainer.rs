use crate::{
    DataPoint,
    layer::Layer,
    loss::{CrossEntropy, Loss, Mse},
    network::Network,
    optim::{Adam, AdamW, Optimizer, Sgd},
};

pub struct Trainer {
    network: Network,
    batch_size: u32,
    param_grad_buffer: Vec<f32>,
    value_buffer: Vec<Vec<f32>>,
    value_grad_buffer: Vec<Vec<f32>>,
    target_buffer: Vec<f32>,
    loss_fn: Box<dyn Loss>,
    optimizer: Box<dyn Optimizer>,
}

impl Trainer {
    fn new(
        network: Network,
        batch_size: u32,
        loss_fn: Box<dyn Loss>,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        let mut value_buffer = Vec::with_capacity(network.layers().len() + 1);
        value_buffer.push(vec![
            0.0;
            (batch_size * network.layers()[0].input_size()) as usize
        ]);
        for layer in network.layers() {
            value_buffer.push(vec![0.0; (batch_size * layer.output_size()) as usize]);
        }

        let num_params = network.num_params();
        Self {
            network: network,
            batch_size,
            param_grad_buffer: vec![0.0; num_params as usize],
            value_buffer: value_buffer.clone(),
            value_grad_buffer: value_buffer.clone(),
            target_buffer: value_buffer.last().unwrap().clone(),
            loss_fn,
            optimizer,
        }
    }

    pub fn network(&self) -> &Network {
        &self.network
    }

    pub fn loss_fn(&self) -> &dyn Loss {
        self.loss_fn.as_ref()
    }

    pub fn run_batch(&mut self, batch: &[DataPoint]) {
        assert!(batch.len() == self.batch_size as usize);

        for (idx, data_pt) in batch.iter().enumerate() {
            self.value_buffer[0][idx * data_pt.input.len()..(idx + 1) * data_pt.input.len()]
                .copy_from_slice(&data_pt.input);
        }

        self.forward_all();

        let output_size = self.network.layers().last().unwrap().output_size();
        for (idx, data_pt) in batch.iter().enumerate() {
            self.target_buffer[idx * output_size as usize..(idx + 1) * output_size as usize]
                .copy_from_slice(&data_pt.target);
        }
        self.backward();
        self.update(batch.len() as u32);
    }

    fn forward_all(&mut self) {
        // value_buffer[0] needs to be pre-filled with all the inputs

        for (idx, layer) in self.network.layers().iter().enumerate() {
            let (left, right) = self.value_buffer.split_at_mut(idx + 1);
            let inputs = &left[idx];
            let outputs = &mut right[0];
            match layer {
                Layer::Dense(dense_layer) => dense_layer.forward(
                    &self.network.param_buffer()[dense_layer.param_buffer_range()],
                    inputs,
                    outputs,
                    self.batch_size,
                ),
                Layer::ReLu(relu_layer) => {
                    relu_layer.forward(inputs, outputs, self.batch_size);
                }
            }
        }
    }

    fn backward(&mut self) {
        self.loss_fn.backward(
            self.value_buffer.last().unwrap(),
            &self.target_buffer,
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
                        self.batch_size,
                    );
                }
                Layer::ReLu(relu_layer) => {
                    relu_layer.backward(
                        output_grads,
                        &self.value_buffer[idx],
                        input_grads,
                        self.batch_size,
                    );
                }
            }
        }
    }

    fn update(&mut self, batch_size: u32) {
        self.optimizer.update(
            &mut self.network.param_buffer_mut(),
            &self.param_grad_buffer,
            batch_size,
        );
    }
}

pub struct TrainerBuilder {
    network: Network,
    batch_size: u32,
    loss_fn: Option<Box<dyn Loss>>,
    optimizer: Option<Box<dyn Optimizer>>,
}

impl TrainerBuilder {
    pub fn new(network: Network) -> Self {
        Self {
            network,
            batch_size: 0,
            optimizer: None,
            loss_fn: None,
        }
    }

    pub fn build(self) -> Trainer {
        Trainer::new(
            self.network,
            self.batch_size,
            self.loss_fn
                .expect("Please set a loss function before building a Trainer"),
            self.optimizer
                .expect("Please set an optimizer before building a Trainer"),
        )
    }

    pub fn batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn mse(mut self) -> Self {
        self.loss_fn = Some(Box::new(Mse::new()));
        self
    }

    pub fn cross_entropy(mut self) -> Self {
        self.loss_fn = Some(Box::new(CrossEntropy::new()));
        self
    }

    pub fn adamw(mut self, lr: f32, lambda: f32) -> Self {
        self.optimizer = Some(Box::new(AdamW::new(lr, lambda, &self.network)));
        self
    }

    pub fn adam(mut self, lr: f32) -> Self {
        self.optimizer = Some(Box::new(Adam::new(lr, &self.network)));
        self
    }

    pub fn sgd(mut self, lr: f32) -> Self {
        self.optimizer = Some(Box::new(Sgd::new(lr)));
        self
    }
}
