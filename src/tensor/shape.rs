use core::assert_eq;
use core::panic;

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::{assert_dim, assert_numel};

pub(crate) const MAX_DIM: usize = 4;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Shape {
    pub(crate) shape: [usize; MAX_DIM],
    pub(crate) strides: [usize; MAX_DIM],
    pub(crate) ndim: usize,
    pub(crate) offset: usize,
}

impl Shape {
    #[inline]
    pub fn new(shape: &[usize]) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        // Right now, we only support row major stride by default
        assert!(!shape.is_empty(), "shape should not be an empty vec");
        assert!(
            shape.len() <= MAX_DIM,
            "shape of len > {MAX_DIM} is not supported as of now"
        );
        assert!(
            !shape.iter().any(|&x| x == 0),
            "{:?} should not contain 0",
            shape
        );

        let strides = Self::get_strides_for_shape(shape);
        let mut shape_arr = [0; MAX_DIM];
        shape_arr
            .iter_mut()
            .zip(shape.iter())
            .for_each(|(dst, &src)| *dst = src);

        Shape {
            shape: shape_arr,
            strides,
            ndim: shape.len(),
            offset: 0,
        }
    }

    #[inline]
    pub(crate) fn get_strides_for_shape(shape: &[usize]) -> [usize; MAX_DIM] {
        let mut strides = [1; MAX_DIM];
        let mut cum_prod = 1;
        let ndim = shape.len();

        strides[..ndim]
            .iter_mut()
            .rev()
            .zip(shape.iter().rev())
            .for_each(|(st, sh)| {
                *st = cum_prod;
                cum_prod *= sh;
            });
        strides[shape.len()..].iter_mut().for_each(|x| *x = 0);
        strides
    }

    #[inline]
    pub fn full(shape: &[usize]) -> Self {
        let mut shape = Self::new(shape);
        shape.strides = [0, 0, 0, 0];
        shape
    }

    #[inline]
    pub(crate) fn from_len(len: usize) -> Self {
        Shape {
            shape: [len, 0, 0, 0],
            strides: [1, 0, 0, 0],
            ndim: 1,
            offset: 0,
        }
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.ndim]
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.strides() == Shape::new(self.shape()).strides()
    }

    #[inline]
    pub fn is_valid_index(&self, index: &[usize]) -> bool {
        !index.is_empty()
            && index.len() <= self.ndim
            && index.iter().zip(self.shape().iter()).all(|(i, s)| i < s)
    }

    pub fn get_buffer_idx(&self, index: &[usize]) -> usize {
        assert_eq!(
            index.len(),
            self.ndim,
            "len of index({}) should be equal to {}",
            index.len(),
            self.ndim,
        );
        self.offset
            + index
                .iter()
                .zip(self.strides().iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>()
    }

    pub fn index_iter(&self) -> TensorIndexIterator {
        TensorIndexIterator::new(self)
    }

    // Removes a dimension from the shape. For eg, let's say we want remove the dimension 1 from
    // the shape [x, y, z]. This method turns the shape [x, y, z] => [x, z] with appropriate strides.
    #[inline]
    pub(crate) fn remove_dim(&self, dim: usize) -> Self {
        assert_dim!(dim, self.ndim);

        let mut shape = self.shape;
        shape[dim..].rotate_left(1);
        shape[3] = 0;

        let mut strides = self.strides;
        strides[dim..].rotate_left(1);
        strides[3] = 0;

        Shape {
            shape,
            strides,
            ndim: self.ndim - 1,
            offset: self.offset,
        }
    }

    // Reduces the given dimension to 1. For eg, let's say we want reduce the dimension 0 from the
    // shape [x, y, z]. This method turns the shape [x, y, z] => [1, y, z] with appropriate
    // strides.
    #[inline]
    pub(crate) fn reduce_dim(&self, dim: usize) -> (Self, Self) {
        assert_dim!(dim, self.ndim);
        let mut reduced_shape = self.shape;
        reduced_shape[dim] = 1;

        let mut reduced_shape = Shape::new(&reduced_shape[..self.ndim]);
        reduced_shape.strides[dim] = 1;

        let mut stride_shape = reduced_shape;
        stride_shape.strides[dim] = 0;
        (reduced_shape, stride_shape)
    }

    pub(crate) fn squeeze(&self) -> Self {
        let mut shape = [0; MAX_DIM];
        let mut strides = [0; MAX_DIM];
        let mut ndim = 0;

        self.shape()
            .iter()
            .zip(self.strides().iter())
            .for_each(|(&dim, &stride)| {
                if dim != 1 {
                    shape[ndim] = dim;
                    strides[ndim] = stride;
                    ndim += 1;
                }
            });

        if ndim == 0 {
            return Self {
                shape: [1, 0, 0, 0],
                strides: [1, 0, 0, 0],
                ndim: 1,
                offset: 0,
            };
        }
        Self {
            shape,
            strides,
            ndim,
            offset: self.offset,
        }
    }

    pub(crate) fn attempt_reshape_without_copying(
        &self,
        new_shape: &[usize],
    ) -> Result<Self, String> {
        // function to check if tensor can be reshaped without copying
        // see https://github.com/numpy/numpy/blob/ac3baf5e229a502b43042c570d4d79e92702669a/numpy/core/src/multiarray/shape.c#L371
        assert_numel!(self.numel(), new_shape.iter().product(), new_shape);
        assert!(
            new_shape.len() <= MAX_DIM,
            "len of new_shape({}) should be less than or equal to {MAX_DIM}",
            new_shape.len()
        );

        // squeeze the current shape, cause axes with dim 1 won't have any effect on the final
        // shape and strides and only adds to complexity.
        let self_squeezed = self.squeeze();

        let old_ndim = self_squeezed.ndim;
        let old_shape = self_squeezed.shape;
        let old_strides = self_squeezed.strides;

        let new_ndim = new_shape.len();
        let new_shape = new_shape.to_vec();
        let mut new_strides = vec![0; new_ndim];

        let (mut new_start, mut new_end) = (0, 1);
        let (mut old_start, mut old_end) = (0, 1);

        // iterate through both new and old shape at least till we exhaust one of them
        while new_start < new_ndim && old_start < old_ndim {
            let mut new_numel = new_shape[new_start];
            let mut old_numel = old_shape[old_start];

            // NOTE: this loop will always stop because we asserted that the numels for both
            // new and old shape are the same at the start of this function.
            // Greedily match the number of elements of both old and new shape
            while new_numel != old_numel {
                if new_numel < old_numel {
                    new_numel *= new_shape[new_end];
                    new_end += 1;
                } else {
                    // here: new_numel > old_numel
                    old_numel *= old_shape[old_end];
                    old_end += 1;
                }
            }

            // check if the "sub-shape" from old shape is contiguous
            // old_end is 1 + the len of old "sub-shape"
            for dim in old_start..old_end - 1 {
                if old_strides[dim] != old_strides[dim + 1] * old_shape[dim + 1] {
                    // not contiguous, need to copy data.
                    return Err(format!(
                        "cannot reshape {:?} to {:?} without copying data",
                        self.shape, new_shape
                    ));
                }
            }

            // "sub-shape" is contiguous, populate strides for new_shape from (new_start..new_end - 1)
            // new_end is 1 + the len of new "sub-shape"
            new_strides[new_end - 1] = old_strides[old_end - 1];
            for dim in (new_start..new_end - 1).rev() {
                new_strides[dim] = new_strides[dim + 1] * new_shape[dim + 1];
            }

            old_start = old_end;
            old_end += 1;
            new_start = new_end;
            new_end += 1;
        }

        let last_stride = new_strides[new_start - 1];
        // we skip through `new_start` items, cause new_start was set to new_end at the end of the
        // previous loop. it can also be new_end - 1 instead of new_start
        // all remaining elems are dims with 1.
        new_strides.iter_mut().skip(new_start).for_each(|x| {
            *x = last_stride;
        });
        let mut new_shape_arr = [0; MAX_DIM];
        new_shape_arr
            .iter_mut()
            .zip(new_shape.iter())
            .for_each(|(dst, &src)| *dst = src);

        let mut new_strides_arr = [0; MAX_DIM];
        new_strides_arr
            .iter_mut()
            .zip(new_strides.iter())
            .for_each(|(dst, &src)| *dst = src);

        Ok(Shape {
            shape: new_shape_arr,
            strides: new_strides_arr,
            ndim: new_shape.len(),
            offset: self.offset,
        })
    }

    pub(crate) fn expand_to(&self, dims: &[usize]) -> Self {
        assert_eq!(
            self.ndim,
            dims.len(),
            "ndims should be equal for both from shape and to shape"
        );
        let mut shape = [0; MAX_DIM];
        let mut strides = [0; MAX_DIM];
        (0..self.ndim).for_each(|i| {
            if self.shape[i] == dims[i] {
                shape[i] = self.shape[i];
                strides[i] = self.strides[i];
            } else if self.shape[i] == 1 {
                shape[i] = dims[i];
                strides[i] = 0;
            } else {
                panic!(
                    "cannot expand shape from {:?} to {:?} at dim {}",
                    self.shape, dims, i
                );
            }
        });
        Shape {
            shape,
            strides,
            ndim: self.ndim,
            offset: self.offset,
        }
    }

    pub(crate) fn expand(&self, dim: usize, to: usize) -> Self {
        assert_dim!(dim, self.ndim);
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
        let mut shape = [0; MAX_DIM];
        let mut strides = [0; MAX_DIM];
        perm_shape.iter().enumerate().for_each(|(i, &from)| {
            shape[i] = self.shape[from];
            strides[i] = self.strides[from];
        });
        Self {
            shape,
            strides,
            ndim: self.ndim,
            offset: self.offset,
        }
    }

    pub(crate) fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        assert_dim!(dim1, dim2, self.ndim);
        let mut new_dims = [0; MAX_DIM];
        new_dims
            .iter_mut()
            .enumerate()
            .for_each(|(i, dst)| *dst = i);
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
        let index = vec![0; shape.ndim];
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
        for dim in (0..self.shape.ndim).rev() {
            self.index[dim] += 1;
            if self.index[dim] < self.shape.shape[dim] {
                break;
            }
            self.index[dim] = 0;
        }
        self.exhausted = self.index.iter().all(|&x| x == 0);
        Some(result)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self::new(&shape)
    }
}

impl From<&[usize]> for Shape {
    fn from(shape: &[usize]) -> Self {
        Self::new(shape)
    }
}

#[cfg(test)]
#[path = "./shape_test.rs"]
mod tests;
