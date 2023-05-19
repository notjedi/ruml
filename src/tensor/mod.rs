mod shape;

pub use self::shape::{Shape, TensorIndexIterator};
use crate::{assert_dim, assert_numel};
use std::{
    fmt::{Debug, Display},
    sync::Arc,
    vec,
};

// https://stackoverflow.com/questions/40929867/how-do-you-abstract-generics-in-nested-rust-types
pub trait Num:
    num_traits::Num + num_traits::cast::NumCast + num_traits::NumAssignOps + PartialOrd + Debug + Copy
{
}

impl<T> Num for T where
    T: num_traits::Num
        + num_traits::cast::NumCast
        + num_traits::NumAssignOps
        + PartialOrd
        + Debug
        + Copy
{
}

#[derive(Eq, PartialEq)]
pub struct Tensor<T: Num> {
    data: Arc<Vec<T>>,
    shape: Shape,
}

impl<T: Num> Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor with shape {:?}", self.shape.shape)
    }
}

impl<T: Num> Debug for Tensor<T> {
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

impl<T: Num> Tensor<T> {
    pub fn new(data: Vec<T>) -> Self {
        let shape = Shape {
            shape: [data.len()].into(),
            strides: [1].into(),
            offset: 0,
        };
        Self {
            data: Arc::new(data),
            shape,
        }
    }

    pub fn arange(len: usize) -> Self {
        let data = (0..len).map(|i| T::from(i).unwrap()).collect::<Vec<_>>();
        Self {
            data: Arc::new(data),
            shape: Shape::from_len(len),
        }
    }

    pub fn zeros(len: usize) -> Self {
        let data = vec![T::zero(); len];
        Self {
            data: Arc::new(data),
            shape: Shape::from_len(len),
        }
    }

    pub fn ones(len: usize) -> Self {
        let data = vec![T::one(); len];
        Self {
            data: Arc::new(data),
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
            data: Arc::clone(&self.data),
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
            data: Arc::clone(&self.data),
            shape,
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Self {
        let shape: Shape = shape.into();
        assert_numel!(self.shape.numel(), shape);
        Tensor {
            data: Arc::clone(&self.data),
            shape,
        }
    }

    pub fn expand(&self, dim: usize, to: usize) -> Self {
        let shape = self.shape.expand(dim, to);
        Self {
            data: Arc::clone(&self.data),
            shape,
        }
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        let shape = self.shape.transpose(dim1, dim2);
        Self {
            data: Arc::clone(&self.data),
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
                    data: Arc::new(sum_buffer),
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

impl<'a, T: Num> IntoIterator for &'a Tensor<T> {
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
pub struct TensorIterator<'a, T: Num> {
    tensor: &'a Tensor<T>,
    index_iter: TensorIndexIterator<'a>,
}

impl<'a, T: Num> Iterator for TensorIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iter
            .next()
            .map(|index| self.tensor.data[self.tensor.shape.get_buffer_idx(&index)])
    }
}

#[derive(Debug)]
pub struct DimIterator<'a, T: Num> {
    tensor: &'a Tensor<T>,
    iter_dim: usize,
    dim_idx: usize,
}

impl<'a, T: Num> Iterator for DimIterator<'a, T> {
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.dim_idx >= self.tensor.shape()[self.iter_dim] {
            return None;
        }
        let mut shape = self.tensor.shape.remove_dim(self.iter_dim);
        shape.offset = self.tensor.stride()[self.iter_dim] * self.dim_idx;
        self.dim_idx += 1;
        let tensor = Tensor {
            data: Arc::clone(&self.tensor.data),
            shape,
        };
        Some(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(Arc::try_unwrap(sum_tensor.data).unwrap(), sum_vec);
    }
}
