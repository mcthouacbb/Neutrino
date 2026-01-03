pub struct ReluLayer {
    size: u32,
}

impl ReluLayer {
    pub fn new(size: u32) -> Self {
        Self { size }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn forward(&self, inputs: &[f32], outputs: &mut [f32], batch_size: u32) {
        assert!(inputs.len() == (batch_size * self.size) as usize);
        assert!(outputs.len() == (batch_size * self.size) as usize);

        for j in 0..(batch_size * self.size) as usize {
            outputs[j] = inputs[j].max(0.0);
        }
    }

    pub fn backward(
        &self,
        output_grads: &[f32],
        inputs: &[f32],
        input_grads: &mut [f32],
        batch_size: u32,
    ) {
        assert!(output_grads.len() == (batch_size * self.size) as usize);
        assert!(inputs.len() == (batch_size * self.size) as usize);
        assert!(input_grads.len() == (batch_size * self.size) as usize);

        input_grads.fill(0.0);

        for j in 0..(self.size * batch_size) as usize {
            if inputs[j] < 0.0 {
                input_grads[j] = 0.0;
            } else {
                input_grads[j] = output_grads[j];
            }
        }
    }
}
