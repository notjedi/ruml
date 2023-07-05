mod shape;
pub use self::shape::{Shape, TensorIndexIterator};

use crate::{
    assert_dim, assert_numel,
    types::{Num, NumFloat, NumInt},
    CACHELINE_ALIGN,
};
use aligned_vec::{avec, AVec};
use alloc::vec;
use alloc::{sync::Arc, vec::Vec};
use core::{
    assert_eq,
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Neg, Sub},
};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

#[derive(Eq, PartialEq)]
pub struct Tensor<T>
where
    T: Num,
{
    pub(crate) data: Arc<AVec<T>>,
    pub(crate) shape: Shape,
}

impl<T> Display for Tensor<T>
where
    T: Num,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Tensor with shape {:?}", self.shape.shape)
    }
}

impl<T> Debug for Tensor<T>
where
    T: Num,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Tensor with shape: {:?} and stride: {:?} and data: {:?}",
            self.shape.shape,
            self.shape.strides,
            self.data.as_slice()
        )
    }
}

macro_rules! impl_ops {
    ($op_trait:ident, $op_fn:ident, $op:tt) => {
        impl<T> $op_trait<&Tensor<T>> for &Tensor<T>
        where
            T: Num,
        {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &Tensor<T>) -> Self::Output {
                self.broadcasted_zip(&rhs, |x, y| x $op y)
            }
        }

        impl<T> $op_trait<&Tensor<T>> for Tensor<T>
        where
            T: Num,
        {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &Tensor<T>) -> Self::Output {
                self.broadcasted_zip(&rhs, |x, y| x $op y)
            }
        }

        impl<T> $op_trait<Tensor<T>> for &Tensor<T>
        where
            T: Num,
        {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: Tensor<T>) -> Self::Output {
                self.broadcasted_zip(&rhs, |x, y| x $op y)
            }
        }

        impl<T> $op_trait<Tensor<T>> for Tensor<T>
        where
            T: Num,
        {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: Tensor<T>) -> Self::Output {
                self.broadcasted_zip(&rhs, |x, y| x $op y)
            }
        }

        impl<T> $op_trait<T> for Tensor<T>
        where
            T: Num,
        {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: T) -> Self::Output {
                let mut add_vec = (*self.data).clone();
                add_vec.iter_mut().for_each(|x| *x = *x $op rhs);
                Tensor {
                    data: Arc::new(add_vec),
                    shape: self.shape.clone(),
                }
            }
        }

        impl<T> $op_trait<T> for &Tensor<T>
        where
            T: Num,
        {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: T) -> Self::Output {
                let mut add_vec = (*self.data).clone();
                add_vec.iter_mut().for_each(|x| *x = *x $op rhs);
                Tensor {
                    data: Arc::new(add_vec),
                    shape: self.shape.clone(),
                }
            }
        }
    };
}

// https://stackoverflow.com/questions/73464666/how-to-implement-stdops-traits-for-multiple-rhs
// https://stackoverflow.com/questions/24594374/how-can-an-operator-be-overloaded-for-different-rhs-types-and-return-values
impl_ops!(Add, add, +);
impl_ops!(Sub, sub, -);
impl_ops!(Mul, mul, *);
impl_ops!(Div, div, /);

impl<T: Num> Neg for Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Tensor<T> {
        let mut neg_vec = (*self.data).clone();
        neg_vec.iter_mut().for_each(|x| *x = -(*x));
        Tensor {
            data: Arc::new(neg_vec),
            shape: self.shape.clone(),
        }
    }
}

impl<T: Num> Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Tensor<T> {
        let mut neg_vec = (*self.data).clone();
        neg_vec.iter_mut().for_each(|x| *x = -(*x));
        Tensor {
            data: Arc::new(neg_vec),
            shape: self.shape.clone(),
        }
    }
}

// TODO: MOVEMENT OPS
// 1. PAD
// 2. SHRINK

// TODO: BINARY OPS
// 1. support broadcast

// TODO: impl<T: Num> std::ops::Add<&Tensor<T>> for &mut Tensor<T> {
// TODO: impl<T: Num> std::ops::Add<Tensor<T>> for &mut Tensor<T> {
// TODO: impl<T: Num> std::ops::Add<T> for &mut Tensor<T>
// TODO: get around to implement mut iter for tensor

// we only support channel last memory format see
// https://pytorch.org/blog/tensor-memory-format-matters for details
// https://ajcr.net/stride-guide-part-1

impl<T> Tensor<T>
where
    T: Num,
{
    pub fn new(data: AVec<T>) -> Self {
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
        let data = AVec::from_iter(CACHELINE_ALIGN, (0..len).map(|i| T::from(i).unwrap()));
        Self {
            data: Arc::new(data),
            shape: Shape::from_len(len),
        }
    }

    pub fn eye(dim: usize) -> Self {
        let mut eye = avec![T::zero(); dim * dim];
        (0..dim).for_each(|i| {
            eye[i * dim + i] = T::one();
        });
        Self {
            data: Arc::new(eye),
            shape: vec![dim; 2].into(),
        }
    }

    pub fn tril(dim: usize) -> Self {
        let mut tril = avec![T::one(); dim * dim];
        (0..dim).for_each(|i| {
            (0..i).for_each(|j| {
                let idx = j * dim + i;
                tril[idx] = T::zero();
            })
        });
        Self {
            data: Arc::new(tril),
            shape: vec![dim; 2].into(),
        }
    }

    pub fn triu(dim: usize) -> Self {
        let mut triu = avec![T::one(); dim * dim];
        (0..dim).for_each(|i| {
            (0..i).for_each(|j| {
                let idx = i * dim + j;
                triu[idx] = T::zero();
            })
        });
        Self {
            data: Arc::new(triu),
            shape: vec![dim; 2].into(),
        }
    }

    pub fn randn<R>(shape: &[usize], rng: &mut R) -> Self
    where
        R: Rng,
        // do i need to include ?Sized for R like this: R: Rng + ?Sized,
        // TODO : what does this bound mean?
        StandardNormal: Distribution<T>,
    {
        let shape: Shape = shape.into();
        let mut data: AVec<T> = AVec::with_capacity(CACHELINE_ALIGN, shape.numel());
        for _ in 0..shape.numel() {
            data.push(rng.sample(StandardNormal));
        }
        Self {
            data: Arc::new(data),
            shape,
        }
    }

    // lazy init of tensor with value
    pub fn full(fill_value: T, shape: &[usize]) -> Self {
        let data = avec![fill_value; 1];
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

    pub fn full_like(fill_value: T, other: &Self) -> Self {
        Self::full(fill_value, &other.shape.shape)
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::full(T::one(), shape)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::full(T::zero(), shape)
    }

    pub fn ones_like(other: &Self) -> Self {
        Self::ones(&other.shape.shape)
    }

    pub fn zeros_like(other: &Self) -> Self {
        Self::zeros(&other.shape.shape)
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape.shape()
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        self.shape.strides()
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous()
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
    pub fn ravel(&self) -> AVec<T> {
        AVec::from_iter(CACHELINE_ALIGN, self.into_iter())
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
    pub fn shallow_clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
        }
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

    pub fn view(&mut self, shape: &[usize]) {
        let shape: Shape = shape.into();
        assert_numel!(self.shape.numel(), shape);
        self.shape = shape;
    }

    #[allow(non_snake_case)]
    pub fn T(&self) -> Self {
        let mut new_shape = self.shape.shape.clone();
        new_shape.reverse();
        self.permute(&new_shape)
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
            return Self {
                data: Arc::clone(&self.data),
                shape,
            };
        }
        let reshape_tensor = self.contiguous();
        reshape_tensor.reshape(&shape)
    }

    pub fn expand_to(&self, dims: &[usize]) -> Self {
        Self {
            data: Arc::clone(&self.data),
            shape: self.shape.expand_to(dims),
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

    pub fn eq(&self, other: &Self) -> Self {
        self.broadcasted_zip(&other, |x, y| if x == y { T::one() } else { T::zero() })
    }

    pub fn nq(&self, other: &Self) -> Self {
        self.broadcasted_zip(&other, |x, y| if x != y { T::one() } else { T::zero() })
    }

    pub fn as_type<S>(&self) -> Tensor<S>
    where
        S: Num,
    {
        // TODO: write tests
        let data = AVec::from_iter(
            CACHELINE_ALIGN,
            self.data.iter().map(|&x| num_traits::cast(x).unwrap()),
        );
        Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        }
    }

    pub fn map(&self, f: impl Fn(T) -> T) -> Self {
        let map_data = AVec::from_iter(CACHELINE_ALIGN, (*self.data).iter().map(|&x| f(x)));
        Self {
            data: Arc::new(map_data),
            shape: self.shape.clone(),
        }
    }

    pub fn abs(&self) -> Self {
        self.map(|x| x.abs())
    }

    pub fn square(&self) -> Self {
        self.map(|x| x * x)
    }

    pub fn relu(&self) -> Self {
        self.map(|x| if x > T::zero() { x } else { T::zero() })
    }

    pub fn zip(&self, other: &Self, f: impl Fn(T, T) -> T) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "shapes {:?} of self and {:?} of other must match",
            self.shape(),
            other.shape()
        );
        let data = AVec::from_iter(
            CACHELINE_ALIGN,
            self.into_iter()
                .zip(other.into_iter())
                .map(|(x, y)| f(x, y)),
        );
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
                .map(|(&x, &y)| core::cmp::max(x, y))
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

    pub fn reduce(&self, default: T, dim: usize, f: impl Fn(T, T) -> T) -> Self {
        assert_dim!(dim, self.shape.ndim());
        let (reduced_shape, stride_shape) = self.shape.reduce_dim(dim);
        let mut reduce_buffer = avec![default; reduced_shape.numel()];
        self.shape.index_iter().for_each(|index| {
            let self_idx = self.shape.get_buffer_idx(&index);
            let stride_idx = stride_shape.get_buffer_idx(&index);
            reduce_buffer[stride_idx] = f(reduce_buffer[stride_idx], self.data[self_idx]);
        });
        Self {
            data: Arc::new(reduce_buffer),
            shape: reduced_shape,
        }
    }

    pub fn sum<S>(&self, dim: S) -> Self
    where
        S: Into<Option<usize>>,
    {
        match dim.into() {
            // TODO: which one should i  use?
            // Some(dim) => self.reduce(T::zero(), dim, std::ops::Add::add),
            Some(dim) => self.reduce(T::zero(), dim, |x, y| x + y),
            None => {
                // BUG: wont' work for non-contiguous tensors
                let sum = avec![self.data.iter().fold(T::zero(), |acc, &x| acc + x)];
                Self::new(sum)
            }
        }
    }

    pub fn max<S>(&self, dim: S) -> Self
    where
        S: Into<Option<usize>>,
    {
        // TODO: write tests
        match dim.into() {
            Some(dim) => self.reduce(T::min_value(), dim, |x, y| if x > y { x } else { y }),
            None => {
                let sum = avec![self.data.iter().fold(T::zero(), |acc, &x| acc + x)];
                Self::new(sum)
            }
        }
    }

    pub fn min<S>(&self, dim: S) -> Self
    where
        S: Into<Option<usize>>,
    {
        // TODO: write tests
        match dim.into() {
            Some(dim) => self.reduce(T::max_value(), dim, |x, y| if x < y { x } else { y }),
            None => {
                let sum = avec![self.data.iter().fold(T::zero(), |acc, &x| acc + x)];
                Self::new(sum)
            }
        }
    }
}

impl<T> Tensor<T>
where
    T: NumFloat,
{
    pub fn exp(&self) -> Self {
        self.map(|x| x.exp())
    }

    pub fn log(&self) -> Self {
        self.map(|x| x.ln())
    }

    pub fn log2(&self) -> Self {
        self.map(|x| x.log2())
    }

    pub fn log10(&self) -> Self {
        self.map(|x| x.log10())
    }

    pub fn powf(&self, pow: T) -> Self {
        self.map(|x| x.powf(pow))
    }

    pub fn powi(&self, pow: i32) -> Self {
        self.map(|x| x.powi(pow))
    }

    pub fn sqrt(&self) -> Self {
        self.map(|x| x.sqrt())
    }

    pub fn tanh(&self) -> Self {
        self.map(|x| x.tan())
    }

    pub fn linspace(start: T, end: T, steps: usize) -> Self {
        // NOTE: this fn would panic if steps == 0; this is intentional cause we do not support
        // empty tensors.
        // Tensor::linspace(3.0, 10.0, 5);
        // [  3.0,   4.75,   6.5,   8.25,  10.0]
        let mut linspace = AVec::with_capacity(CACHELINE_ALIGN, steps);
        let dx = (end - start) / (T::from(steps).unwrap() - T::one());
        (0..steps).for_each(|i| {
            linspace.push(start + dx * T::from(i).unwrap());
        });
        Self {
            data: Arc::new(linspace),
            shape: vec![steps; 1].into(),
        }
    }
}

impl<T> Tensor<T>
where
    T: NumInt,
{
    pub fn pow(&self, pow: u32) -> Self {
        self.map(|x| x.pow(pow))
    }
}

impl<'a, T> IntoIterator for &'a Tensor<T>
where
    T: Num,
{
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
pub struct TensorIterator<'a, T>
where
    T: Num,
{
    tensor: &'a Tensor<T>,
    index_iter: TensorIndexIterator<'a>,
}

impl<'a, T> Iterator for TensorIterator<'a, T>
where
    T: Num,
{
    type Item = T;

    // TODO: write size_hint and count functions
    fn next(&mut self) -> Option<Self::Item> {
        self.index_iter
            .next()
            .map(|index| self.tensor.data[self.tensor.shape.get_buffer_idx(&index)])
    }
}

#[derive(Debug)]
pub struct DimIterator<'a, T>
where
    T: Num,
{
    tensor: &'a Tensor<T>,
    iter_dim: usize,
    dim_idx: usize,
}

impl<'a, T> Iterator for DimIterator<'a, T>
where
    T: Num,
{
    type Item = Tensor<T>;

    // TODO: write size_hint and count functions
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
#[path = "./tensor_test.rs"]
mod tests;
