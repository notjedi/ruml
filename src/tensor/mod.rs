#[derive(Debug)]
pub struct Shape {
    dim: Box<[usize]>,
    strides: Box<[usize]>,
}

impl Shape {
    pub fn new(dim: Box<[usize]>) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        // Right now, we only support row major stride by default
        let mut strides = vec![1; dim.len()].into_boxed_slice();
        let stride_it = strides.iter_mut().rev();
        let mut cum_prod = 1;
        for (st, dim) in stride_it.zip(dim.iter().rev()) {
            *st = cum_prod;
            cum_prod *= dim;
        }

        Shape { dim, strides }
    }

    pub fn ndim(&self) -> usize {
        self.dim.len()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dim: Vec<usize>) -> Self {
        Self::new(dim.into_boxed_slice())
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Tensor<T> {
    data: T,
    dim: Shape,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Borrow;

    #[test]
    fn test_shape() {
        let shape: Shape = vec![2, 2, 5].into();
        assert_eq!(shape.dim.borrow(), [2, 2, 5]);
        assert_eq!(shape.strides.borrow(), [10, 5, 1]);
    }
}
