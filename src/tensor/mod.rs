use std::ops::Add;

#[derive(Debug, PartialEq, Eq)]
pub struct Shape {
    shape: Box<[usize]>,
    strides: Box<[usize]>,
}

impl Shape {
    pub fn new(shape: Box<[usize]>) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        // Right now, we only support row major stride by default
        // TODO: return error if any of the elems in dim is 0
        let mut strides = vec![1; shape.len()].into_boxed_slice();
        let stride_it = strides.iter_mut().rev();
        let mut cum_prod = 1;
        for (st, sh) in stride_it.zip(shape.iter().rev()) {
            *st = cum_prod;
            cum_prod *= sh;
        }

        Shape { shape, strides }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    // TODO: make this a field so we don't have to calculate every time?
    pub fn numel(&self) -> usize {
        self.shape.iter().fold(1, |acc, &i| acc * i)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self::new(shape.into_boxed_slice())
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
// TODO: make data a lifetime field?
// assert that the type of tensor is always a number
pub struct Tensor<T: Add<Output = T> + Default + Copy> {
    // TODO: should i take a pointer to the data?
    data: Box<[T]>,
    shape: Shape,
}

impl<T: Add<Output = T> + Default + Copy> Tensor<T> {
    pub fn from_data(data: Box<[T]>) -> Self {
        let shape = Shape {
            shape: [data.len()].into(),
            strides: [1].into(),
        };
        Self { data, shape }
    }

    pub fn from_shape(shape: Shape, data: Box<[T]>) -> Self {
        // TODO: return error if len(data) != shape.numel()
        assert_eq!(data.len(), shape.numel());
        Self { data, shape }
    }

    // TODO: make the argument Into<Shape> and calculate stride ourselves
    pub fn view(&mut self, shape: Shape) {
        assert_eq!(shape.numel(), self.shape.numel());
        self.shape = shape;
    }

    // TODO: this will be moved to backend in the future
    // TODO: what should be the type of axis?
    pub fn sum(&self, dim: Option<usize>) -> Self {
        // NOTE: need not check if axis is < 0 as far as the type is unsigned
        match dim {
            Some(dim) => {
                assert!(dim < self.shape.ndim());
                // let new_dim = Shape
                // TODO: change the implementation
                let start = T::default();
                let sum = Box::new([self.data.iter().fold(start, |acc, x| acc + *x)]);
                Self::from_data(sum)
            }
            None => {
                let start = T::default();
                let sum = Box::new([self.data.iter().fold(start, |acc, x| acc + *x)]);
                Self::from_data(sum)
            }
        }
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
        assert_eq!(shape.shape.borrow(), [2, 2, 5]);
        assert_eq!(shape.strides.borrow(), [10, 5, 1]);
    }

    #[test]
    fn test_tensor() {
        let shape: Shape = vec![2, 2, 2].into();
        let data = Box::new([1; 2 * 2 * 2]);
        let tensor: Tensor<i32> = Tensor::from_shape(shape, data);
        let sum_tensor = tensor.sum(None);
        let sum_tensor_check: Tensor<i32> = Tensor::from_data([8].into());
        dbg!(&sum_tensor);
        // assert!(false);
        assert_eq!(sum_tensor, sum_tensor_check);

        // TODO: assert that this is actually equal to a 3d tensor
        assert_eq!(tensor.data.borrow(), [1, 1, 1, 1, 1, 1, 1, 1]);
    }
}
