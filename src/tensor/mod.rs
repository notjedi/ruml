use crate::{assert_dim, assert_numel};
use std::{
    fmt::{Debug, Display},
    rc::Rc,
    vec,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape {
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}

impl Shape {
    #[inline]
    pub fn new(shape: Vec<usize>) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        // Right now, we only support row major stride by default
        assert!(!shape.is_empty(), "shape should not be an empty vec");
        assert!(
            !shape.iter().any(|&x| x == 0),
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
        Shape {
            shape,
            strides,
            offset: 0,
        }
    }

    #[inline]
    pub(crate) fn from_len(len: usize) -> Self {
        Shape {
            shape: [len].into(),
            strides: [1].into(),
            offset: 0,
        }
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn stride(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline]
    pub fn is_valid_index(&self, index: &[usize]) -> bool {
        !index.is_empty()
            && index.len() <= self.shape.len()
            && index.iter().zip(self.shape.iter()).all(|(i, s)| i < s)
    }

    pub fn get_buffer_idx(&self, index: &[usize]) -> usize {
        self.offset
            + index
                .iter()
                .zip(self.strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>()
    }

    pub fn index_iter(&self) -> TensorIndexIterator {
        TensorIndexIterator::new(self)
    }

    // Removes a dimension from the shape. For eg, let's say we want remove the dimension 1 from
    // the shape [x, y, z]. This method turns the shape [x, y, z] => [x, z] with appropriate strides.
    pub(crate) fn remove_dim(&self, dim: usize) -> Self {
        assert_dim!(dim, self.ndim());
        let mut shape = self.clone();
        shape.shape.remove(dim);
        shape.strides.remove(dim);
        shape
    }

    // Reduces the given dimension to 1. For eg, let's say we want reduce the dimension 0 from the
    // shape [x, y, z]. This method turns the shape [x, y, z] => [1, y, z] with appropriate
    // strides.
    pub(crate) fn reduce_dim(&self, dim: usize) -> Self {
        assert_dim!(dim, self.ndim());
        let mut reduced_shape = self.shape.clone();
        reduced_shape[dim] = 1;
        let mut reduced_shape = Shape::new(reduced_shape);
        // TODO: it is okay for a stride to be 0?
        reduced_shape.strides[dim] = 0;
        reduced_shape
    }

    pub(crate) fn squeeze(&self) -> Self {
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
        if shape.is_empty() {
            return Self {
                shape: vec![1],
                strides: vec![1],
                offset: 0,
            };
        }
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    pub(crate) fn expand(&self, dim: usize, to: usize) -> Self {
        assert_dim!(dim, self.ndim());
        assert!(
            self.shape[dim] == 1,
            "cannot expand shape {:?} at dim {}.",
            self.shape,
            dim
        );
        let mut expand_shape = self.clone();
        expand_shape.shape[dim] = to;
        expand_shape.strides[dim] = 0;
        expand_shape
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
            !perm_shape.iter().any(|&x| x >= self.ndim()),
            "All dimensions should be less than {}",
            self.ndim()
        );
        let mut shape = Vec::with_capacity(self.ndim());
        let mut strides = Vec::with_capacity(self.ndim());
        perm_shape.iter().for_each(|&i| {
            shape.push(self.shape[i]);
            strides.push(self.strides[i]);
        });
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    pub(crate) fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        assert_dim!(dim1, dim2, self.ndim());
        let mut new_dims = (0..self.ndim()).collect::<Vec<usize>>();
        new_dims.swap(dim1, dim2);
        self.permute(&new_dims)
    }
}

#[derive(Debug)]
pub struct TensorIndexIterator<'a> {
    shape: &'a Shape,
    index: Vec<usize>,
    exhausted: bool,
}

impl<'a> TensorIndexIterator<'a> {
    pub fn new(shape: &'a Shape) -> Self {
        let index = vec![0; shape.ndim()];
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

impl From<&[usize]> for Shape {
    fn from(shape: &[usize]) -> Self {
        Self::new(shape.to_vec())
    }
}

#[derive(Eq, PartialEq)]
pub struct Tensor<T: crate::Num> {
    data: Rc<Vec<T>>,
    shape: Shape,
}

impl<T: crate::Num> Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor with shape {:?}", self.shape.shape)
    }
}

impl<T: crate::Num> Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor with shape: {:?} and stride: {:?} and data: {:?}",
            self.shape.shape,
            self.shape.strides,
            self.data.as_slice()
        )
    }
}

// TODO: get around to implement mut iter for tensor
// TODO: naive reduce functions

/// we only support channel last memory format
/// see <https://pytorch.org/blog/tensor-memory-format-matters> for details
/// <https://ajcr.net/stride-guide-part-1>

impl<T: crate::Num> Tensor<T> {
    pub fn new(data: Vec<T>) -> Self {
        let shape = Shape {
            shape: [data.len()].into(),
            strides: [1].into(),
            offset: 0,
        };
        Self {
            data: Rc::new(data),
            shape,
        }
    }

    pub fn arange(len: usize) -> Self {
        let data = (0..len).map(|i| T::from(i).unwrap()).collect::<Vec<_>>();
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

    pub fn ones(len: usize) -> Self {
        let data = vec![T::one(); len];
        Self {
            data: Rc::new(data),
            shape: Shape::from_len(len),
        }
    }

    #[inline]
    pub fn flatten(&self) -> Self {
        self.reshape(&[self.shape.numel()])
    }

    #[inline]
    pub fn squeeze(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
            shape: self.shape.squeeze(),
        }
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape.shape()
    }

    #[inline]
    pub fn stride(&self) -> &[usize] {
        self.shape.stride()
    }

    pub fn view(&mut self, shape: &[usize]) {
        let shape: Shape = shape.into();
        assert_numel!(self.shape.numel(), shape);
        self.shape = shape;
    }

    pub fn permute(&self, dims: &[usize]) -> Self {
        let shape = self.shape.permute(dims);
        Self {
            data: Rc::clone(&self.data),
            shape,
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Self {
        let shape: Shape = shape.into();
        assert_numel!(self.shape.numel(), shape);
        Tensor {
            data: Rc::clone(&self.data),
            shape,
        }
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        let shape = self.shape.transpose(dim1, dim2);
        Self {
            data: Rc::clone(&self.data),
            shape,
        }
    }

    pub fn expand(&self, dim: usize, to: usize) -> Self {
        let shape = self.shape.expand(dim, to);
        Self {
            data: Rc::clone(&self.data),
            shape,
        }
    }

    pub fn dim_iter(&self, dim: usize) -> DimIterator<T> {
        assert_dim!(dim, self.shape.ndim());
        DimIterator {
            tensor: self,
            iter_dim: dim,
            dim_idx: 0,
        }
    }

    // https://hoverbear.org/blog/optional-arguments
    pub fn sum<S>(&self, dim: S) -> Self
    where
        S: Into<Option<usize>>,
    {
        match dim.into() {
            Some(dim) => {
                assert_dim!(dim, self.shape.ndim());
                let reduced_shape = self.shape.reduce_dim(dim);
                let mut sum_buffer = vec![T::zero(); reduced_shape.numel()];
                for index in self.shape.index_iter() {
                    sum_buffer[reduced_shape.get_buffer_idx(&index)] +=
                        self.data[self.shape.get_buffer_idx(&index)];
                }
                Self {
                    data: Rc::new(sum_buffer),
                    shape: reduced_shape,
                }
            }
            None => {
                let sum = [self.data.iter().fold(T::zero(), |acc, &x| acc + x)].into();
                Self::new(sum)
            }
        }
    }
}

impl<'a, T: crate::Num> IntoIterator for &'a Tensor<T> {
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
pub struct TensorIterator<'a, T: crate::Num> {
    tensor: &'a Tensor<T>,
    index_iter: TensorIndexIterator<'a>,
}

impl<'a, T: crate::Num> Iterator for TensorIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iter
            .next()
            .map(|index| self.tensor.data[self.tensor.shape.get_buffer_idx(&index)])
    }
}

#[derive(Debug)]
pub struct DimIterator<'a, T: crate::Num> {
    tensor: &'a Tensor<T>,
    iter_dim: usize,
    dim_idx: usize,
}

impl<'a, T: crate::Num> Iterator for DimIterator<'a, T> {
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.dim_idx >= self.tensor.shape()[self.iter_dim] {
            return None;
        }
        let mut shape = self.tensor.shape.remove_dim(self.iter_dim);
        shape.offset = self.tensor.stride()[self.iter_dim] * self.dim_idx;
        self.dim_idx += 1;
        let tensor = Tensor {
            data: Rc::clone(&self.tensor.data),
            shape,
        };
        Some(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let shape_vec = vec![3, 2, 5, 1];
        let shape: Shape = shape_vec.clone().into();

        assert_eq!(shape.ndim(), 4);
        assert_eq!(shape.shape(), &shape_vec);
        assert_eq!(shape.stride(), &[10, 5, 1, 1]);
        assert_eq!(shape.numel(), shape_vec.iter().product());
        assert_eq!(shape.is_valid_index(&[2, 1, 3, 0]), true);
        assert_eq!(shape.is_valid_index(&[10, 3, 0, 10]), false);
    }

    #[test]
    fn test_shape_ops() {
        // TODO: add tests for other ops like transpose, reduce_dim, etc
        let shape_vec = vec![3, 2, 5, 1];
        let shape: Shape = shape_vec.clone().into();

        let remove_shape = shape.remove_dim(1);
        assert_eq!(remove_shape.shape(), &[3, 5, 1]);
        assert_eq!(remove_shape.stride(), &[10, 1, 1]);

        let squeeze_shape = shape.squeeze();
        assert_eq!(squeeze_shape.shape(), &[3, 2, 5]);
        assert_eq!(squeeze_shape.stride(), &[10, 5, 1]);

        let perm_shape = shape.permute(&[3, 2, 1, 0]);
        assert_eq!(perm_shape.shape(), &[1, 5, 2, 3]);
        assert_eq!(perm_shape.stride(), &[1, 1, 5, 10]);

        let trans_shape = shape.transpose(0, 3);
        assert_eq!(trans_shape.shape(), &[1, 2, 5, 3]);
        assert_eq!(trans_shape.stride(), &[1, 5, 1, 10]);
    }

    #[test]
    fn test_tensor() {
        let shape_vec = vec![2, 2, 2];
        let shape: Shape = shape_vec.clone().into();

        let ones_tensor: Tensor<f32> = Tensor::ones(shape.numel()).reshape(&shape_vec);
        assert_eq!(ones_tensor.shape(), shape.shape());
        assert_eq!(ones_tensor.data.len(), shape.numel());
        assert_eq!(
            ones_tensor.data.as_slice(),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "elements don't match for the Tensor::ones tensor"
        );

        let zeros_tensor: Tensor<f32> = Tensor::zeros(shape.numel()).reshape(&shape_vec);
        assert_eq!(zeros_tensor.data.len(), shape.numel());
        assert_eq!(zeros_tensor.shape(), shape.shape());
        assert_eq!(
            zeros_tensor.data.as_slice(),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "elements don't match for the Tensor::zeros tensor"
        );

        let arange_tensor: Tensor<f32> = Tensor::arange(shape.numel()).reshape(&shape_vec);
        assert_eq!(arange_tensor.data.len(), shape.numel());
        assert_eq!(arange_tensor.shape(), shape.shape());
        assert_eq!(
            arange_tensor.data.as_slice(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "elements don't match for the Tensor::arange tensor"
        );
    }

    #[test]
    fn test_tensor_iter() {
        // TODO: test tensor iter after shape ops like transpose, expand, etc
        let shape_vec = vec![2, 2, 2];
        let shape: Shape = shape_vec.clone().into();

        let ones_tensor: Tensor<f32> = Tensor::ones(shape.numel()).reshape(&shape_vec);
        ones_tensor
            .into_iter()
            .zip(ones_tensor.data.iter())
            .enumerate()
            .for_each(|(i, (iter, &vec))| {
                assert_eq!(iter, vec, "values differ at {i}");
            });
    }

    #[test]
    fn test_tensor_dim_iter() {
        let shape_vec = vec![1, 2, 3];
        let shape: Shape = shape_vec.clone().into();
        let arange_tensor: Tensor<f32> = Tensor::arange(shape.numel()).reshape(&shape_vec);

        // should return the following elements [0, 1, 2, 3, 4, 5]
        for dim_tensor in arange_tensor.dim_iter(0) {
            // dbg!(dim_tensor.into_iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                dim_tensor.into_iter().collect::<Vec<_>>().as_slice(),
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            );
        }

        // should return the following elements [[0, 1, 2], [3, 4, 5]]
        let mut num = 0.0;
        for dim_tensor in arange_tensor.dim_iter(1) {
            // dbg!(dim_tensor.into_iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                dim_tensor.into_iter().collect::<Vec<_>>().as_slice(),
                vec![num, num + 1.0, num + 2.0]
            );
            num += 3.0;
        }

        // should return the following elements [[0, 3], [1, 4], [2, 5]]
        for (i, dim_tensor) in arange_tensor.dim_iter(2).enumerate() {
            // dbg!(dim_tensor.into_iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                dim_tensor.into_iter().collect::<Vec<_>>().as_slice(),
                vec![i as f32, i as f32 + 3.0]
            );
        }
    }

    #[test]
    fn test_tensor_ops() {
        let shape_vec = vec![3, 4, 5];
        let shape: Shape = shape_vec.clone().into();
        let tensor: Tensor<f32> = Tensor::arange(shape.numel()).reshape(&shape_vec);

        let sum_tensor = tensor.sum(None);
        let sum = (shape.numel() * (shape.numel() - 1) / 2) as f32;
        let sum_tensor_check: Tensor<f32> = Tensor::new([sum].into());
        assert_eq!(sum_tensor, sum_tensor_check);

        let sum_tensor = tensor.sum(2);
        // import numpy as np; shape = [3, 4, 5]; np.arange(np.prod(shape), dtype=np.float32).reshape(shape).sum(axis=2).flatten()
        let sum_vec: Vec<f32> = vec![
            10.0, 35.0, 60.0, 85.0, 110.0, 135.0, 160.0, 185.0, 210.0, 235.0, 260.0, 285.0,
        ];
        assert_eq!(sum_tensor.shape(), &[3, 4, 1]);
        assert_eq!(Rc::try_unwrap(sum_tensor.data).unwrap(), sum_vec);
    }
}
