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

    pub fn forward(&self, inputs: &[f32], outputs: &mut [f32]) {
        assert!(inputs.len() == self.size as usize);
        assert!(outputs.len() == self.size as usize);

        for i in 0..self.size as usize {
            outputs[i] = inputs[i].max(0.0);
        }
    }

    pub fn backward(&self, output_grads: &[f32], inputs: &[f32], input_grads: &mut [f32]) {
        assert!(output_grads.len() == self.size as usize);
        assert!(inputs.len() == self.size as usize);
        assert!(input_grads.len() == self.size as usize);

        for i in 0..self.size as usize {
            if inputs[i] < 0.0 {
                input_grads[i] = 0.0;
            } else {
                input_grads[i] = output_grads[i];
            }
        }
    }
}
