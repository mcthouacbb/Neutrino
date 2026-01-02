mod dense_layer;
mod relu_layer;

pub use dense_layer::*;
pub use relu_layer::*;

pub enum Layer {
    ReLu(ReluLayer),
    Dense(DenseLayer),
}

impl Layer {
    pub fn input_size(&self) -> u32 {
        match self {
            Self::Dense(dense_layer) => dense_layer.input_size(),
            Self::ReLu(relu_layer) => relu_layer.size(),
        }
    }

    pub fn output_size(&self) -> u32 {
        match self {
            Self::Dense(dense_layer) => dense_layer.output_size(),
            Self::ReLu(relu_layer) => relu_layer.size(),
        }
    }
}
