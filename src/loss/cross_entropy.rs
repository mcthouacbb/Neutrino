use crate::loss::Loss;

pub struct CrossEntropy {}

impl CrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for CrossEntropy {
    fn forward(&self, inputs: &[f32], targets: &[f32]) -> f32 {
        let mut max = inputs[0];
        for i in 1..inputs.len() {
            max = max.max(inputs[i]);
        }

        let mut exp_sum = 0.0;
        for input in inputs {
            exp_sum += (input - max).exp();
        }

        let log_exp_sum = exp_sum.ln();

        let mut loss = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            loss -= target * (input - max - log_exp_sum);
        }
        loss
    }

    fn backward(&self, inputs: &[f32], targets: &[f32], grads: &mut [f32]) {
        for (i, v) in grads.iter_mut().enumerate() {
            *v = inputs[i] - targets[i];
        }
    }
}
