use std::{ops::Add, rc::Rc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
}

impl Shape {
    pub fn new(shape: Vec<usize>) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        // Right now, we only support row major stride by default
        // TODO: return error if any of the elems in dim is 0
        let mut strides = vec![1; shape.len()];
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

    pub fn remove_dim(&mut self) {}
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self::new(shape)
    }
}

pub trait RawTensor {
    type Elem: num_traits::Float;
}

#[derive(Debug, PartialEq, Eq)]
pub struct Tensor<T: num_traits::Float> {
    data: Rc<Vec<T>>,
    shape: Shape,
}

// TODO: make data a lifetime field?
// assert that the type of tensor is always a number
/// we only support channel last memory format
/// see https://pytorch.org/blog/tensor-memory-format-matters for details

impl<T: num_traits::Float> Tensor<T> {
    pub fn from_data(data: Vec<T>) -> Self {
        let shape = Shape {
            shape: [data.len()].into(),
            strides: [1].into(),
        };
        Self {
            data: Rc::new(data),
            shape,
        }
    }

    pub fn from_shape(shape: Shape, data: Vec<T>) -> Self {
        // TODO: return error if len(data) != shape.numel()
        assert_eq!(data.len(), shape.numel());
        Self {
            data: Rc::new(data),
            shape,
        }
    }

    // TODO: make the argument Into<Shape> and calculate stride ourselves
    pub fn view(&mut self, shape: Shape) {
        assert_eq!(shape.numel(), self.shape.numel());
        self.shape = shape;
    }

    // TODO: this will be moved to backend in the future
    // TODO: what should be the type of axis?
    pub fn sum(&self, dim: Option<usize>) -> Self {
        match dim {
            Some(dim) => {
                // NOTE: need not check if axis is < 0 as far as the type is unsigned
                assert!(dim < self.shape.ndim());
                todo!();
            }
            None => {
                let sum = [self.data.iter().fold(T::zero(), |acc, x| acc + *x)].into();
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
        let data = vec![1.0; 2 * 2 * 2];
        let tensor: Tensor<f32> = Tensor::from_shape(shape, data);
        let sum_tensor = tensor.sum(None);
        let sum_tensor_check: Tensor<f32> = Tensor::from_data([8.0].into());
        dbg!(&sum_tensor);
        assert_eq!(sum_tensor, sum_tensor_check);

        // assert!(false);
        // TODO: assert that this is actually equal to a 3d tensor
        assert_eq!(
            tensor.data.as_slice(),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }
}
