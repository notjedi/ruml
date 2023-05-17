use std::{
    fmt::{Debug, Display},
    rc::Rc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl Shape {
    // TODO: always inline few methods
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

    pub(crate) fn from_len(len: usize) -> Self {
        Shape {
            shape: [len].into(),
            strides: [1].into(),
        }
    }

    pub(crate) fn empty() -> Self {
        Self {
            shape: vec![],
            strides: vec![],
        }
    }

    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline(always)]
    pub fn stride(&self) -> &[usize] {
        &self.strides
    }

    #[inline(always)]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[inline(always)]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline(always)]
    pub(crate) fn is_valid_index(&self, index: &[usize]) -> bool {
        // TODO: should the len of index be equal to self.shape.len()?
        !index.is_empty()
            // && !self.shape.is_empty() // rendered redundant by the next check
            && index.len() <= self.shape.len()
            && index.iter().zip(self.shape.iter()).all(|(i, s)| i < s)
    }

    pub(crate) fn get_buffer_idx(&self, index: Vec<usize>) -> usize {
        // TODO: should i assert that the length of both the index vec and that of the strides vec
        // is same?
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum::<usize>()
    }

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

#[derive(Debug)]
pub(crate) struct TensorIndexIterator<'a> {
    shape: &'a Shape,
    index: Vec<usize>,
    exhausted: bool,
}

impl<'a> TensorIndexIterator<'a> {
    pub(crate) fn new(shape: &'a Shape) -> Self {
        let index = vec![0; shape.ndim()];
        // TODO: the function call to check if it's a valid index is not really needed and we can
        // just set it to true here, as long as we don't support scalar tensors
        let exhausted = !shape.is_valid_index(&index);
        Self {
            shape,
            index,
            exhausted,
        }
    }
}

impl<'a> Iterator for TensorIndexIterator<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let result = self.index.clone();
        for dim in (0..self.shape.ndim()).rev() {
            self.index[dim] += 1;
            if self.index[dim] < self.shape.shape[dim] {
                break;
            }
            self.index[dim] = 0;
        }
        self.exhausted = self.index.iter().all(|&x| x == 0);
        return Some(result);
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self::new(shape)
    }
}

// use T as num_traits::Num?
#[derive(PartialEq, Eq)]
pub struct Tensor<T: num_traits::Float> {
    data: Rc<Vec<T>>,
    shape: Shape,
}

impl<T: num_traits::Float> Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor with shape {:?}", self.shape.shape)
    }
}

impl<T: num_traits::Float> Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor with shape: {:?} and stride: {:?}",
            self.shape.shape, self.shape.strides
        )
    }
}

// TODO: support scalar Tensor
// TODO: how do we really want Eq and PartialEq to work
// TODO: add axis_iter method to iter through each axis of tensor
// TOOD: replace asserts w Result type?
/// we only support channel last memory format
/// see https://pytorch.org/blog/tensor-memory-format-matters for details

impl<T: num_traits::Float + num_traits::NumAssignOps> Tensor<T> {
    // TODO: change arg type to Into<Shape>?
    pub fn new(data: Vec<T>) -> Self {
        let shape = Shape {
            shape: [data.len()].into(),
            strides: [1].into(),
        };
        Self {
            data: Rc::new(data),
            shape,
        }
    }

    pub fn ones(len: usize) -> Self {
        let data = vec![T::one(); len];
        Self {
            data: Rc::new(data),
            shape: Shape::from_len(len),
        }
    }

    pub fn zeros(len: usize) -> Self {
        let data = vec![T::zero(); len];
        Self {
            data: Rc::new(data),
            shape: Shape::from_len(len),
        }
    }

    pub fn arange(len: usize) -> Self {
        let data = (0..len).map(|i| T::from(i).unwrap()).collect::<Vec<_>>();
        Self {
            data: Rc::new(data),
            shape: Shape::from_len(len),
        }
    }

    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        self.shape.shape()
    }

    // TODO: make the argument Into<Shape> and calculate stride ourselves
    pub fn view(&mut self, shape: Shape) {
        // TODO: create macro or helpers for asserting numels
        assert_eq!(
            self.shape.numel(),
            shape.numel(),
            "shape {:?} is invalid for input of size {}.",
            shape.shape,
            self.shape.numel()
        );
        self.shape = shape;
    }

    #[inline(always)]
    pub fn flatten(&self) -> Self {
        let shape = Shape {
            shape: vec![self.shape.numel()],
            strides: vec![1],
        };
        self.reshape(shape)
    }

    pub fn reshape(&self, shape: Shape) -> Self {
        assert_eq!(
            self.shape.numel(),
            shape.numel(),
            "shape {:?} is invalid for input of size {}.",
            shape.shape,
            self.shape.numel()
        );
        Tensor {
            data: Rc::clone(&self.data),
            shape,
        }
    }

    pub fn permute(&self, dims: &[usize]) -> Self {
        let shape = self.shape.permute(dims);
        Self {
            data: Rc::clone(&self.data),
            shape,
        }
    }

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Self {
        let shape = self.shape.transpose(dim_1, dim_2);
        Self {
            data: Rc::clone(&self.data),
            shape,
        }
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
                Self::new(sum)
            }
        }
    }
}

impl<'a, T: num_traits::Float + num_traits::NumAssignOps> IntoIterator for &'a Tensor<T> {
    type Item = T;
    type IntoIter = TensorIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIterator {
            tensor: self,
            index_iter: TensorIndexIterator::new(&self.shape),
        }
    }
}

#[derive(Debug)]
pub struct TensorIterator<'a, T: num_traits::Float + num_traits::NumAssignOps> {
    tensor: &'a Tensor<T>,
    index_iter: TensorIndexIterator<'a>,
}

impl<'a, T: num_traits::Float + num_traits::NumAssignOps> Iterator for TensorIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iter
            .next()
            .map(|index| self.tensor.data[self.tensor.shape.get_buffer_idx(index)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let shape_vec = vec![3, 2, 5, 1];
        let shape: Shape = shape_vec.clone().into();

        assert_eq!(shape.shape(), &shape_vec);
        assert_eq!(shape.stride(), &[10, 5, 1, 1]);
        assert_eq!(shape.ndim(), 4);
        assert_eq!(shape.numel(), shape_vec.iter().product());
        assert_eq!(shape.is_valid_index(&[2, 1, 3, 0]), true);
        assert_eq!(shape.is_valid_index(&[10, 3, 0, 10]), false);

        let remove_shape = shape.remove_dim(1);
        assert_eq!(remove_shape.shape(), &[3, 5, 1]);
        assert_eq!(remove_shape.stride(), &[5, 1, 1]);

        let squeeze_shape = shape.squeeze();
        assert_eq!(squeeze_shape.shape(), &[3, 2, 5]);
        assert_eq!(squeeze_shape.stride(), &[10, 5, 1]);

        let perm_shape = shape.permute(&[3, 2, 1, 0]);
        assert_eq!(perm_shape.shape(), &[1, 5, 2, 3]);
        assert_eq!(perm_shape.stride(), &[1, 1, 5, 10]);
    }

    #[test]
    #[should_panic]
    fn test_shape_remove_dim() {
        let shape_vec = vec![3, 2, 5, 1];
        let shape: Shape = shape_vec.clone().into();
        shape.remove_dim(4);
    }

    #[test]
    fn test_tensor() {
        let shape: Shape = vec![2, 2, 2].into();
        let ones_tensor: Tensor<f32> = Tensor::ones(shape.numel()).reshape(shape.clone());
        assert_eq!(ones_tensor.shape(), shape.shape());
        assert_eq!(ones_tensor.data.len(), shape.numel());
        assert_eq!(
            ones_tensor.data.as_slice(),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "elements don't match for the Tensor::ones tensor"
        );

        let zeros_tensor: Tensor<f32> = Tensor::zeros(shape.numel()).reshape(shape.clone());
        assert_eq!(zeros_tensor.data.len(), shape.numel());
        assert_eq!(zeros_tensor.shape(), shape.shape());
        assert_eq!(
            zeros_tensor.data.as_slice(),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "elements don't match for the Tensor::zeros tensor"
        );

        let arange_tensor: Tensor<f32> = Tensor::arange(shape.numel()).reshape(shape.clone());
        assert_eq!(arange_tensor.data.len(), shape.numel());
        assert_eq!(arange_tensor.shape(), shape.shape());
        assert_eq!(
            arange_tensor.data.as_slice(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "elements don't match for the Tensor::arange tensor"
        );

        // test iterator
        ones_tensor
            .into_iter()
            .zip(ones_tensor.data.iter())
            .enumerate()
            .for_each(|(i, (iter, &vec))| {
                assert_eq!(iter, vec, "values differ at {i}");
            });
    }

    #[test]
    fn test_binary_ops() {
        let shape: Shape = vec![2, 2, 2].into();
        let data = vec![1.0; 2 * 2 * 2];
        let tensor: Tensor<f32> = Tensor::new(data).reshape(shape);

        let sum_tensor = tensor.sum(None);
        let sum_tensor_check: Tensor<f32> = Tensor::new([8.0].into());
        assert_eq!(sum_tensor, sum_tensor_check);
    }
}
