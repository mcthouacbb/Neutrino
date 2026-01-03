use std::ops::Range;

use rand::Rng;

pub struct DenseLayer {
    input_size: u32,
    output_size: u32,
    buffer_offset: u32,
}

impl DenseLayer {
    pub fn new(input_size: u32, output_size: u32, buffer_offset: u32) -> Self {
        Self {
            input_size,
            output_size,
            buffer_offset,
        }
    }

    pub fn input_size(&self) -> u32 {
        self.input_size
    }

    pub fn output_size(&self) -> u32 {
        self.output_size
    }

    fn num_weights(&self) -> u32 {
        self.input_size * self.output_size
    }

    fn num_biases(&self) -> u32 {
        self.output_size
    }

    pub fn num_params(&self) -> u32 {
        self.num_weights() + self.num_biases()
    }

    pub fn param_buffer_range(&self) -> Range<usize> {
        self.buffer_offset as usize..(self.buffer_offset + self.num_params()) as usize
    }

    pub fn init_rand(&self, param_buffer: &mut [f32]) {
        assert!(param_buffer.len() == self.num_params() as usize);

        let bound = (6.0 / self.input_size as f32).sqrt();
        for weight in &mut param_buffer[0..self.num_weights() as usize] {
            *weight = rand::rng().random_range(-bound..=bound);
        }
    }

    pub fn forward(
        &self,
        param_buffer: &[f32],
        inputs: &[f32],
        outputs: &mut [f32],
        batch_size: u32,
    ) {
        assert!(param_buffer.len() == self.num_params() as usize);
        assert!(inputs.len() == (self.input_size * batch_size) as usize);
        assert!(outputs.len() == (self.output_size * batch_size) as usize);

        let (weights, biases) = param_buffer.split_at(self.num_weights() as usize);

        for i in 0..batch_size as usize {
            for j in 0..self.output_size as usize {
                outputs[i * self.output_size as usize + j] = biases[j];
                for k in 0..self.input_size as usize {
                    outputs[i * self.output_size as usize + j] += inputs
                        [i * self.input_size as usize + k]
                        * weights[j * self.input_size as usize + k];
                }
            }
        }
    }

    pub fn backward(
        &self,
        param_buffer: &[f32],
        output_grads: &[f32],
        inputs: &[f32],
        result_grads: &mut [f32],
        input_grads: &mut [f32],
        batch_size: u32,
    ) {
        assert!(param_buffer.len() == self.num_params() as usize);
        assert!(output_grads.len() == (batch_size * self.output_size) as usize);
        assert!(inputs.len() == (batch_size * self.input_size) as usize);
        assert!(result_grads.len() == self.num_params() as usize);
        assert!(input_grads.len() == (batch_size * self.input_size) as usize);

        let (weights, biases) = param_buffer.split_at(self.num_weights() as usize);

        let (weight_grads, bias_grads) = result_grads.split_at_mut(self.num_weights() as usize);

        bias_grads.fill(0.0);
        weight_grads.fill(0.0);
        input_grads.fill(0.0);

        // bias gradients
        for i in 0..batch_size as usize {
            for j in 0..self.output_size as usize {
                bias_grads[j] += output_grads[i * self.output_size as usize + j];
            }
        }

        // weight gradients
        for i in 0..batch_size as usize {
            for j in 0..self.output_size as usize {
                for k in 0..self.input_size as usize {
                    weight_grads[j * self.input_size as usize + k] += inputs
                        [i * self.input_size as usize + k]
                        * output_grads[i * self.output_size as usize + j];
                }
            }
        }

        for i in 0..batch_size as usize {
            for j in 0..self.output_size as usize {
                for k in 0..self.input_size as usize {
                    input_grads[i * self.input_size as usize + k] += output_grads
                        [i * self.output_size as usize + j]
                        * weights[j * self.input_size as usize + k];
                }
            }
        }
    }
}
