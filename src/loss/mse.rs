use crate::loss::Loss;

pub struct Mse {}

impl Mse {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for Mse {
    fn forward(&self, inputs: &[f32], targets: &[f32]) -> f32 {
        assert!(inputs.len() == targets.len());

        let mut result = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let diff = *input - *target;
            result += diff * diff;
        }
        result
    }

    fn backward(&self, inputs: &[f32], targets: &[f32], grads: &mut [f32]) {
        assert!(inputs.len() == targets.len());
        assert!(inputs.len() == grads.len());

        for (i, v) in grads.iter_mut().enumerate() {
            let diff = inputs[i] - targets[i];
            *v = 2.0 * diff;
        }
    }
}
