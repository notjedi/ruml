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
        // TODO: return error if any of the elems in dim is 0
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

    pub fn numel(&self) -> usize {
        self.dim.iter().fold(1, |acc, &i| acc * i)
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
    // TODO: should i take a pointer to the data?
    data: Box<[T]>,
    dim: Shape,
}

impl<T> Tensor<T> {
    pub fn from_shape(dim: Shape, data: Box<[T]>) -> Self {
        // TODO: return error if len(data) != shape.numel()
        assert_eq!(data.len(), dim.numel());
        Self { data, dim }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Borrow;

    #[test]
    fn test_shape() {
        let shape: Shape = vec![2, 2, 5].into();
        assert_eq!(shape.numel(), 2 * 2 * 5);
        assert_eq!(shape.dim.borrow(), [2, 2, 5]);
        assert_eq!(shape.strides.borrow(), [10, 5, 1]);
    }

    #[test]
    fn test_tensor() {
        let shape: Shape = vec![2, 2, 2].into();
        let data = Box::new([1; 2 * 2 * 2]);
        let tensor: Tensor<i32> = Tensor::from_shape(shape, data);

        // TODO: assert that this is actually equal to a 3d tensor
        assert_eq!(tensor.data.borrow(), [1, 1, 1, 1, 1, 1, 1, 1]);
    }
}
