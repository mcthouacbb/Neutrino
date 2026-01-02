mod dense_layer;
mod relu_layer;

pub use dense_layer::*;
pub use relu_layer::*;

pub enum Layer {
    ReLu(ReluLayer),
    Dense(DenseLayer),
}
