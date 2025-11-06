use std::ops;

#[derive(Clone, Debug)]
pub struct Matrix {
    elems: Vec<f32>,
    width: u32,
    height: u32,
}

impl Matrix {
    pub fn zeros(width: u32, height: u32) -> Self {
        Self {
            elems: vec![0.0; (width * height) as usize],
            width: width,
            height: height,
        }
    }

    pub fn elems(&self) -> &Vec<f32> {
        &self.elems
    }

    pub fn elems_mut(&mut self) -> &mut Vec<f32> {
        &mut self.elems
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

impl ops::Index<u32> for Matrix {
    type Output = [f32];
    fn index(&self, index: u32) -> &Self::Output {
        let begin = index * self.width();
        let end = begin + self.width();
        &self.elems[begin as usize..end as usize]
    }
}

impl ops::IndexMut<u32> for Matrix {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        let begin = index * self.width();
        let end = begin + self.width();
        &mut self.elems[begin as usize..end as usize]
    }
}
