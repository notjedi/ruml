mod shape;

pub use self::shape::{Shape, TensorIndexIterator};
use crate::{assert_dim, assert_numel};
use std::{
    assert_eq,
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

// TODO: impl<T: Num> std::ops::Add<&Tensor<T>> for &mut Tensor<T> {
// TODO: impl<T: Num> std::ops::Add<Tensor<T>> for &mut Tensor<T> {
// TODO: impl<T: Num> std::ops::Add<T> for &mut Tensor<T>

impl<T: Num> std::ops::Add<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        self.broadcasted_zip(&rhs, |x, y| x + y)
    }
}

impl<T: Num> std::ops::Add<&Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        self.broadcasted_zip(&rhs, |x, y| x + y)
    }
}

impl<T: Num> std::ops::Add<Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Tensor<T>) -> Tensor<T> {
        self.broadcasted_zip(&rhs, |x, y| x + y)
    }
}

impl<T: Num> std::ops::Add<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Tensor<T>) -> Self::Output {
        self.broadcasted_zip(&rhs, |x, y| x + y)
    }
}

impl<T: Num> std::ops::Add<T> for Tensor<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let mut add_vec = (*self.data).clone();
        add_vec.iter_mut().for_each(|x| *x += rhs);
        Self {
            data: Arc::new(add_vec),
            shape: self.shape.clone(),
        }
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

    // lazy init of tensor with value
    pub fn full(value: T, shape: &[usize]) -> Self {
        let data = vec![value; 1];
        let shape = Shape {
            shape: shape.to_vec(),
            strides: vec![0; shape.len()],
            offset: 0,
        };
        Self {
            data: Arc::new(data),
            shape,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::full(T::zero(), shape)
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::full(T::one(), shape)
    }

    #[inline]
    pub fn flatten(&self) -> Self {
        if self.shape.is_contiguous() {
            // BUG(FIXED): let's say i expand a tensor from shape (3, 1) -> (3, 3).
            // then self.shape.numel() would be = 9. but the len of the actual data buffer would be 3.
            // so just copy data to new buffer if the actual buffer len and shape.numel don't match.
            return self.reshape(&[self.shape.numel()]);
        }
        self.contiguous().reshape(&[self.shape.numel()])
    }

    #[inline]
    pub fn shallow_clone(&self) -> Self {
        Self {
            // https://stackoverflow.com/a/55750742
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
        }
    }

    #[inline]
    pub fn clone(&self) -> Self {
        Self {
            // https://stackoverflow.com/a/55750742
            data: Arc::new((*self.data).clone()),
            shape: self.shape.clone(),
        }
    }

    #[inline]
    pub fn ravel(&self) -> Vec<T> {
        self.into_iter().collect::<Vec<T>>()
    }

    #[inline]
    pub fn squeeze(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            shape: self.shape.squeeze(),
        }
    }

    #[inline]
    pub fn contiguous(&self) -> Self {
        if self.shape.is_contiguous() {
            // NOTE: torch only copies data if data buffer is not contiguous
            // so we follow torch
            return Self {
                data: Arc::clone(&self.data),
                shape: self.shape.clone(),
            };
        }
        Self {
            data: Arc::new(self.ravel()),
            shape: Shape::new(&self.shape.shape),
        }
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape.shape()
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        self.shape.strides()
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
        assert_numel!(self.shape.numel(), shape.iter().product(), shape);
        if let Ok(shape) = self.shape.attempt_reshape_without_copying(shape) {
            return Tensor {
                data: Arc::clone(&self.data),
                shape,
            };
        }
        let reshape_tensor = self.contiguous();
        reshape_tensor.reshape(&shape)
    }

    pub fn expand(&self, dim: usize, to: usize) -> Self {
        let shape = self.shape.expand(dim, to);
        Self {
            data: Arc::clone(&self.data),
            shape,
        }
    }

    pub fn expand_to(&self, dims: &[usize]) -> Self {
        Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.expand_to(dims),
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

    pub fn zip(&self, other: &Self, f: impl Fn(T, T) -> T) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "shapes {:?} of self and {:?} of other must match",
            self.shape(),
            other.shape()
        );
        let data = self
            .into_iter()
            .zip(other.into_iter())
            .map(|(x, y)| f(x, y))
            .collect();
        Self {
            data: Arc::new(data),
            shape: Shape::new(&self.shape.shape),
        }
    }

    pub fn broadcasted_zip(&self, other: &Self, f: impl Fn(T, T) -> T) -> Self {
        if self.shape() == other.shape() {
            return self.zip(other, f);
        }

        if self.shape.ndim() == other.shape.ndim() {
            let new_shape = self
                .shape()
                .iter()
                .zip(other.shape())
                .map(|(&x, &y)| std::cmp::max(x, y))
                .collect::<Vec<usize>>();
            let expanded_self = self.expand_to(&new_shape);
            let expanded_other = other.expand_to(&new_shape);
            return expanded_self.zip(&expanded_other, f);
        }

        let ones_to_add = self.shape.ndim().abs_diff(other.shape.ndim());
        let mut new_shape = vec![1; ones_to_add];

        if self.shape.ndim() < other.shape.ndim() {
            new_shape.extend_from_slice(self.shape());
            return self.reshape(&new_shape).broadcasted_zip(&other, f);
        } else {
            // here self.shape.ndim() > other.shape.ndim()
            new_shape.extend_from_slice(other.shape());
            other.reshape(&new_shape).broadcasted_zip(&self, f)
        }
    }

    pub fn sum<S>(&self, dim: S) -> Self
    where
        S: Into<Option<usize>>,
    {
        match dim.into() {
            Some(dim) => {
                assert_dim!(dim, self.shape.ndim());
                let mut reduced_shape = self.shape.reduce_dim(dim);
                let mut sum_buffer = vec![T::zero(); reduced_shape.numel()];
                for index in self.shape.index_iter() {
                    sum_buffer[reduced_shape.get_buffer_idx(&index)] +=
                        self.data[self.shape.get_buffer_idx(&index)];
                }
                // TODO: do i really need to change the stride here?
                reduced_shape.strides[dim] = 1;
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
        shape.offset = self.tensor.strides()[self.iter_dim] * self.dim_idx;
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

        let ones_tensor: Tensor<f32> = Tensor::ones(&shape_vec);
        assert_eq!(ones_tensor.shape(), shape.shape());
        assert_eq!(
            ones_tensor.ravel(),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "elements don't match for the Tensor::ones tensor"
        );

        let zeros_tensor: Tensor<f32> = Tensor::zeros(&shape_vec);
        assert_eq!(zeros_tensor.shape(), shape.shape());
        assert_eq!(
            zeros_tensor.ravel(),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "elements don't match for the Tensor::zeros tensor"
        );

        let arange_tensor: Tensor<f32> = Tensor::arange(shape.numel()).reshape(&shape_vec);
        assert_eq!(arange_tensor.data.len(), shape.numel());
        assert_eq!(arange_tensor.shape(), shape.shape());
        assert_eq!(
            arange_tensor.ravel(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "elements don't match for the Tensor::arange tensor"
        );
    }

    #[test]
    fn test_tensor_iter() {
        // TODO: test tensor iter after shape ops like transpose, expand, etc
        let shape_vec = vec![2, 2, 2];

        let ones_tensor: Tensor<f32> = Tensor::ones(&shape_vec);
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
        let add_shape: Shape = vec![2, 2, 2].into();
        let add_tensor = Tensor::arange(add_shape.numel()).reshape(&add_shape.shape) + 5.0;
        assert_eq!(
            add_tensor.ravel(),
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "elements don't match for the Tensor::arange tensor"
        );
    }

    // #[test]
    // fn test_tensor_shape_ops() {
    //     let shape: Shape = vec![2, 1, 3].into();

    //     let tensor: Tensor<f32> = Tensor::arange(shape.numel())
    //         .reshape(&shape.shape)
    //         .expand(1, 4);
    //     let tensor: Vec<_> = tensor.flatten();
    //     TODO: add tests for flatten after expanding
    //     // assert!(false);
    // }

    #[test]
    fn test_tensor_mlops() {
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

    #[test]
    fn test_tensor_broadcast() {
        let x = Tensor::<f32>::zeros(&[1, 3]);
        let y = Tensor::<f32>::ones(&[3, 1]);
        let broadcast_tensor = x + y;
        assert_eq!(broadcast_tensor.shape(), &[3, 3]);
        assert_eq!(broadcast_tensor.ravel(), vec![1.0; 9]);

        let x = Tensor::<f32>::zeros(&[5, 3, 1]);
        let y = Tensor::<f32>::ones(&[3, 1]);
        let broadcast_tensor = &x + &y;
        assert_eq!(broadcast_tensor.shape(), &[5, 3, 1]);
        assert_eq!(broadcast_tensor.ravel(), vec![1.0; 15]);

        let broadcast_tensor = &y + &x;
        assert_eq!(broadcast_tensor.shape(), &[5, 3, 1]);
        assert_eq!(broadcast_tensor.ravel(), vec![1.0; 15]);
    }
}
