use crate::{
    layer::{DenseLayer, Layer, ReluLayer},
    loss::{Loss, Mse},
    optim::Optimizer,
};

pub struct Network {
    param_buffer: Vec<f32>,
    value_buffer: Vec<Vec<f32>>,
    param_grad_buffer: Vec<f32>,
    value_grad_buffer: Vec<Vec<f32>>,
    layers: Vec<Layer>,
    loss_fn: Box<dyn Loss>,
}

impl Network {
    fn new(num_params: u32, layers: Vec<Layer>) -> Self {
        let mut value_buffer = Vec::with_capacity(layers.len() + 1);
        value_buffer.push(vec![0.0; layers[0].input_size() as usize]);
        for layer in &layers {
            value_buffer.push(vec![0.0; layer.output_size() as usize]);
        }
        let result = Self {
            param_buffer: vec![0.0; num_params as usize],
            value_buffer: value_buffer.clone(),
            param_grad_buffer: vec![0.0; num_params as usize],
            value_grad_buffer: value_buffer,
            layers,
            loss_fn: Box::new(Mse::new()),
        };
        result
    }

    fn forward_all(&mut self, inputs: &[f32]) {
        self.value_buffer[0].copy_from_slice(inputs);

        for (idx, layer) in self.layers.iter().enumerate() {
            let (left, right) = self.value_buffer.split_at_mut(idx + 1);
            let inputs = &left[idx];
            let outputs = &mut right[0];
            match layer {
                Layer::Dense(dense_layer) => dense_layer.forward(
                    &self.param_buffer[dense_layer.param_buffer_range()],
                    inputs,
                    outputs,
                ),
                Layer::ReLu(relu_layer) => {
                    relu_layer.forward(inputs, outputs);
                }
            }
        }
    }

    fn forward_all_loss(&mut self, inputs: &[f32], targets: &[f32]) -> f32 {
        self.forward_all(inputs);
        let loss = self
            .loss_fn
            .forward(self.value_buffer.last().unwrap(), targets);
        loss
    }

    pub fn forward_inference(&mut self, inputs: &[f32], targets: &[f32]) -> (Vec<f32>, f32) {
        let loss = self.forward_all_loss(inputs, targets);
        (self.value_buffer.last().unwrap().clone(), loss)
    }

    pub fn backward(&mut self, inputs: &[f32], targets: &[f32]) {
        self.forward_all(inputs);

        self.loss_fn.backward(
            self.value_buffer.last().unwrap(),
            targets,
            self.value_grad_buffer.last_mut().unwrap(),
        );
        for (idx, layer) in self.layers.iter().enumerate().rev() {
            let (left, right) = self.value_grad_buffer.split_at_mut(idx + 1);
            let output_grads = &right[0];
            let input_grads = &mut left[idx];
            match layer {
                Layer::Dense(dense_layer) => {
                    let layer_params = &self.param_buffer[dense_layer.param_buffer_range()];
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

    pub fn update(&mut self, optim: &mut dyn Optimizer, batch_size: u32) {
        optim.update(&mut self.param_buffer, &self.param_grad_buffer, batch_size);
        self.param_grad_buffer.fill(0.0);
    }

    pub fn init_rand(&mut self) {
        for layer in &mut self.layers {
            match layer {
                Layer::Dense(dense_layer) => {
                    dense_layer.init_rand(&mut self.param_buffer[dense_layer.param_buffer_range()])
                }
                _ => {}
            }
        }
    }

    pub fn num_params(&self) -> u32 {
        self.param_buffer.len() as u32
    }
}

pub struct NetworkBuilder {
    input_size: u32,
    num_params: u32,
    layers: Vec<Layer>,
}

impl NetworkBuilder {
    pub fn new(input_size: u32) -> Self {
        Self {
            input_size,
            num_params: 0,
            layers: Vec::new(),
        }
    }

    pub fn add_dense_layer(&mut self, output_size: u32) {
        let dense_layer = DenseLayer::new(self.next_input_size(), output_size, self.num_params);
        self.num_params += dense_layer.num_params();
        self.add_layer(Layer::Dense(dense_layer));
    }

    pub fn add_relu(&mut self) {
        self.add_layer(Layer::ReLu(ReluLayer::new(self.next_input_size())));
    }

    pub fn build(self) -> Network {
        Network::new(self.num_params, self.layers)
    }

    fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    fn next_input_size(&self) -> u32 {
        if let Some(last_layer) = self.layers.last() {
            last_layer.output_size()
        } else {
            self.input_size
        }
    }
}
