use crate::layer::{DenseLayer, Layer, ReluLayer};

pub struct Network {
    param_buffer: Vec<f32>,
    layers: Vec<Layer>,
}

impl Network {
    fn new(num_params: u32, layers: Vec<Layer>) -> Self {
        let result = Self {
            param_buffer: vec![0.0; num_params as usize],
            layers,
        };
        result
    }

    pub fn forward_inference(&self, inputs: &[f32]) -> Vec<f32> {
        let mut input_buffer = inputs.to_vec();
        let mut output_buffer = Vec::new();
        for layer in &self.layers {
            match layer {
                Layer::Dense(dense_layer) => {
                    output_buffer.resize(dense_layer.output_size() as usize, 0.0);
                    dense_layer.forward(
                        &self.param_buffer[dense_layer.param_buffer_range()],
                        &input_buffer,
                        &mut output_buffer,
                    );

                    input_buffer.resize(output_buffer.len(), 0.0);
                    input_buffer.copy_from_slice(&output_buffer);
                }
                Layer::ReLu(relu_layer) => {
                    output_buffer.resize(relu_layer.size() as usize, 0.0);
                    relu_layer.forward(&input_buffer, &mut output_buffer);

                    input_buffer.resize(output_buffer.len(), 0.0);
                    input_buffer.copy_from_slice(&output_buffer);
                }
            }
        }
        output_buffer
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

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn param_buffer_mut(&mut self) -> &mut [f32] {
        &mut self.param_buffer
    }

    pub fn param_buffer(&self) -> &[f32] {
        &self.param_buffer
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
