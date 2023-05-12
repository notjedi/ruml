use std::{
    fmt::{Debug, Display},
    rc::Rc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

// TODO: always inline few methods
impl Shape {
    pub fn new(shape: Vec<usize>) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        // Right now, we only support row major stride by default
        assert!(!shape.is_empty(), "shape should not be an empty vec");
        assert!(
            !shape.iter().any(|x| *x == 0),
            "{:?} should not contain 0",
            shape
        );

        let mut strides = vec![1; shape.len()];
        let mut cum_prod = 1;
        strides
            .iter_mut()
            .rev()
            .zip(shape.iter().rev())
            .for_each(|(st, sh)| {
                *st = cum_prod;
                cum_prod *= sh;
            });
        Shape { shape, strides }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    // TODO: should i make this inplace?
    // Removes a dimension from the shape. For eg, let's say we want remove the dimension 1 from
    // the shape [x, y, z]. This method turns the shape [x, y, z] => [x, z] with appropriate strides.
    pub(crate) fn remove_dim(&self, dim: usize) -> Self {
        assert!(
            dim < self.ndim(),
            "{} should be within the range of 0 <= dim < {}",
            dim,
            self.ndim()
        );
        let mut shape = self.shape.clone();
        shape.remove(dim);
        Shape::new(shape)
    }

    pub(crate) fn empty() -> Self {
        Self {
            shape: vec![],
            strides: vec![],
        }
    }

    pub(crate) fn is_valid_index(&self, index: &[usize]) -> bool {
        // TODO: should the len of index be equal to self.shape.len()?
        !index.is_empty()
            && index.len() <= self.shape.len()
            && index.iter().zip(self.shape.iter()).all(|(i, s)| i < s)
    }

    pub(crate) fn squeeze(&self) -> Self {
        // TODO: what if the shape is [1, 1, 1] and when squeezing the shape would be empty vec
        let mut shape = Vec::with_capacity(self.ndim());
        let mut strides = Vec::with_capacity(self.ndim());
        self.shape
            .iter()
            .zip(self.strides.iter())
            .for_each(|(&dim, &stride)| {
                if dim != 1 {
                    shape.push(dim);
                    strides.push(stride);
                }
            });
        Self { shape, strides }
    }

    pub(crate) fn permute(&self, perm_shape: &[usize]) -> Self {
        // a clever way to check for duplicate elements, exploiting the fact that the elements
        // should be consecutive https://github.com/kurtschelfthout/tensorken/blob/main/src/shape_strider.rs#L213
        assert_eq!(
            (perm_shape.len() * (perm_shape.len() - 1)) / 2,
            perm_shape.iter().sum(),
            "all dims must be specified exactly once"
        );
        assert!(
            perm_shape.iter().all(|x| *x < self.ndim()),
            "All dimensions should be less than {}",
            self.ndim()
        );
        let mut shape = Vec::with_capacity(self.ndim());
        let mut strides = Vec::with_capacity(self.ndim());
        perm_shape.iter().for_each(|i| {
            shape.push(self.shape[*i]);
            strides.push(self.strides[*i]);
        });
        Self { shape, strides }
    }

    pub(crate) fn transpose(&self, dim_1: usize, dim_2: usize) -> Self {
        assert!(
            dim_1 < self.ndim() && dim_2 < self.ndim(),
            "both dim_1({}) and dim_2({}) should be less than {}",
            dim_1,
            dim_2,
            self.ndim()
        );
        let mut new_dims = (0..self.ndim()).collect::<Vec<usize>>();
        new_dims.swap(dim_1, dim_2);
        self.permute(&new_dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self::new(shape)
    }
}

pub trait RawTensor {
    type Elem: num_traits::Float;
}

// use T as num_traits::Num?
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
        assert!(
            data.len() == shape.numel(),
            "Number of elements should be same in data buffer({}) and shape({}).",
            data.len(),
            shape.numel()
        );
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
