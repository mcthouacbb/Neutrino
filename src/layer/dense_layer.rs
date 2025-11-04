use crate::{matrix::Matrix, vector::Vector};


pub struct DenseLayer {
    input_size: u32,
    output_size: u32,
    weights: Matrix,
    biases: Vector,
}

impl DenseLayer {
    fn new(input_size: u32, output_size: u32) -> Self {
        Self {
            input_size: input_size,
            output_size: output_size,
            weights: Matrix::zeros(output_size, input_size),
            biases: Vector::zeros(output_size)
        }
    }

    fn input_size(&self) -> u32 {
        self.input_size
    }

    fn output_size(&self) -> u32 {
        self.output_size
    }

    fn forward(&self, inputs: &Vector) -> Vector {
        let mut result = self.biases.clone();
        for i in 0..self.output_size {
            let row = &self.weights[i];
            for j in 0..self.input_size {
                result[i] += inputs[j] * row[j as usize];
            }
        }
        result
    }
}
