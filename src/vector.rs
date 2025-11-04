use std::ops;

#[derive(Clone, Debug)]
pub struct Vector(Vec<f32>);

impl Vector {
    fn zeros(len: u32) -> Self {
        Self(vec![0.0; len as usize])
    }

    fn len(&self) -> u32 {
        self.0.len() as u32
    }
}

impl ops::Index<u32> for Vector {
    type Output = f32;
    fn index(&self, index: u32) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl ops::IndexMut<u32> for Vector {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

impl ops::AddAssign<Vector> for Vector {
    fn add_assign(&mut self, rhs: Vector) {
        assert!(self.len() == rhs.len());
        for i in 0..self.len() {
            self[i] += rhs[i];
        }
    }
}

impl ops::Add<Vector> for Vector {
    type Output = Vector;
    fn add(self, rhs: Vector) -> Self::Output {
        assert!(self.len() == rhs.len());
        let mut result = self.clone();
        result += rhs;
        result
    }
}

impl ops::SubAssign<Vector> for Vector {
    fn sub_assign(&mut self, rhs: Vector) {
        assert!(self.len() == rhs.len());
        for i in 0..self.len() {
            self[i] -= rhs[i];
        }
    }
}

impl ops::Sub<Vector> for Vector {
    type Output = Vector;
    fn sub(self, rhs: Vector) -> Self::Output {
        assert!(self.len() == rhs.len());
        let mut result = self.clone();
        result -= rhs;
        result
    }
}
