use std::ops::{Index, IndexMut};

// more shapes TBD
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: [u32; 4],
    order: u32,
}

impl Shape {
    pub fn scalar() -> Self {
        Self { dims: [1; 4], order: 0 }
    }

    pub fn vector(size: u32) -> Self {
        let mut result = Self::scalar();
        result.dims[result.order as usize] = size;
        result.order += 1;

        result
    }

    pub fn matrix(size0: u32, size1: u32) -> Shape {
        let mut result = Self::vector(size0);
        result.dims[result.order as usize] = size1;
        result.order += 1;

        result
    }

    pub fn order(&self) -> u32 {
        self.order
    }

    pub fn dim(&self, n: u32) -> u32 {
        assert!(n < self.order());
        self.dims[n as usize]
    }

    pub fn elems(&self) -> u32 {
        self.dims.iter().product()
    }

    pub fn flat_index(&self, indices: &[u32]) -> u32 {
        assert!(indices.len() as u32 == self.order());
        let mut result = 0;
        for i in 0..self.order() {
            result *= self.dim(i);
            result += indices[i as usize];
        }
        result
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    elems: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    pub fn zeros(shape: Shape) -> Self {
        Self {
            elems: vec![0.0; shape.elems() as usize],
            shape,
        }
    }

    pub fn elems(&self) -> &[f32] {
        &self.elems
    }

    pub fn elems_mut(&mut self) -> &mut [f32] {
        &mut self.elems
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

impl Index<u32> for Tensor {
    type Output = f32;

    fn index(&self, index: u32) -> &Self::Output {
        assert!(self.shape.order() == 1);
        &self.elems[index as usize]
    }
}

impl IndexMut<u32> for Tensor {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        assert!(self.shape.order() == 1);
        &mut self.elems[index as usize]
    }
}

impl Index<(u32, u32)> for Tensor {
    type Output = f32;

    fn index(&self, index: (u32, u32)) -> &Self::Output {
        &self.elems[self.shape.flat_index(&[index.0, index.1]) as usize]
    }
}

impl IndexMut<(u32, u32)> for Tensor {
    fn index_mut(&mut self, index: (u32, u32)) -> &mut Self::Output {
        &mut self.elems[self.shape.flat_index(&[index.0, index.1]) as usize]
    }
}

#[cfg(test)]
mod test {
    use crate::tensor::Shape;

    #[test]
    fn shape() {
        let shape = Shape::scalar();
        assert_eq!(shape.order(), 0);
        assert_eq!(shape.elems(), 1);
        assert_eq!(shape.flat_index(&[]), 0);

        let shape = Shape::vector(42);
        assert_eq!(shape.order(), 1);
        assert_eq!(shape.dim(0), 42);
        assert_eq!(shape.elems(), 42);
        assert_eq!(shape.flat_index(&[8]), 8);

        let shape = Shape::matrix(42, 55);
        assert_eq!(shape.order(), 2);
        assert_eq!(shape.dim(0), 42);
        assert_eq!(shape.dim(1), 55);
        assert_eq!(shape.elems(), 42 * 55);
        assert_eq!(shape.flat_index(&[0, 8]), 8);
        assert_eq!(shape.flat_index(&[1, 8]), 55 + 8);
        assert_eq!(shape.flat_index(&[15, 12]), 15 * 55 + 12);
    }
}
