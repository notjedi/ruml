use core::panic;
use std::assert_eq;

use crate::{assert_dim, assert_numel};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape {
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}

impl Shape {
    #[inline]
    pub fn new(shape: &[usize]) -> Self {
        // TODO: accept &[usize] as arg
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
            shape: shape.to_vec(),
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
    pub fn strides(&self) -> &[usize] {
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
    pub fn is_contiguous(&self) -> bool {
        self.strides() == Shape::new(&self.shape).strides()
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
    #[inline]
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
    #[inline]
    pub(crate) fn reduce_dim(&self, dim: usize) -> (Self, Self) {
        assert_dim!(dim, self.ndim());
        let mut reduced_shape = self.shape.clone();
        reduced_shape[dim] = 1;

        let mut reduced_shape = Shape::new(&reduced_shape);
        reduced_shape.strides[dim] = 1;

        let mut stride_shape = reduced_shape.clone();
        stride_shape.strides[dim] = 0;
        (reduced_shape, stride_shape)
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

        // squeeze the current shape, cause axes with dim 1 won't have any effect on the final
        // shape and strides and only adds to complexity.
        let self_squeezed = self.squeeze();

        let old_ndim = self_squeezed.ndim();
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

        Ok(Shape {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    pub(crate) fn expand_to(&self, dims: &[usize]) -> Self {
        assert_eq!(
            self.ndim(),
            dims.len(),
            "ndims should be equal for both from shape and to shape"
        );
        let mut shape = Vec::with_capacity(self.ndim());
        let mut strides = Vec::with_capacity(self.ndim());
        (0..self.ndim()).for_each(|i| {
            if self.shape[i] == dims[i] {
                shape.push(dims[i]);
                strides.push(self.strides[i]);
            } else if self.shape[i] == 1 {
                shape.push(dims[i]);
                strides.push(0);
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
            offset: self.offset,
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
            offset: self.offset,
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
        Self::new(&shape)
    }
}

impl From<&[usize]> for Shape {
    fn from(shape: &[usize]) -> Self {
        Self::new(shape)
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
        assert_eq!(shape.strides(), &[10, 5, 1, 1]);
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
        assert_eq!(remove_shape.strides(), &[10, 1, 1]);

        let squeeze_shape = shape.squeeze();
        assert_eq!(squeeze_shape.shape(), &[3, 2, 5]);
        assert_eq!(squeeze_shape.strides(), &[10, 5, 1]);

        let perm_shape = shape.permute(&[3, 2, 1, 0]);
        assert_eq!(perm_shape.shape(), &[1, 5, 2, 3]);
        assert_eq!(perm_shape.strides(), &[1, 1, 5, 10]);

        let trans_shape = shape.transpose(0, 3);
        assert_eq!(trans_shape.shape(), &[1, 2, 5, 3]);
        assert_eq!(trans_shape.strides(), &[1, 5, 1, 10]);
    }

    #[test]
    fn test_attempt_reshape_without_copying() {
        // normal shape = contiguous
        let shape = Shape {
            shape: vec![4, 3, 2],
            strides: vec![6, 2, 1],
            offset: 0,
        };
        let attempt_reshape = shape.attempt_reshape_without_copying(&[4, 1, 3, 2]);
        assert_eq!(attempt_reshape.unwrap().strides, &[6, 6, 2, 1]);

        // normal shape = contiguous
        let shape = Shape {
            shape: vec![3, 27],
            strides: vec![27, 1],
            offset: 0,
        };
        let attempt_reshape = shape.attempt_reshape_without_copying(&[3, 3, 3, 3]);
        assert_eq!(attempt_reshape.unwrap().strides, &[27, 9, 3, 1]);

        // normal shape = contiguous, with trailing 1's in new_shape
        let shape = Shape {
            shape: vec![3, 27],
            strides: vec![27, 1],
            offset: 0,
        };
        let attempt_reshape = shape.attempt_reshape_without_copying(&[3, 3, 3, 3, 1]);
        assert_eq!(attempt_reshape.unwrap().strides, &[27, 9, 3, 1, 1]);

        // expanded at dim 0
        let shape = Shape {
            shape: vec![8, 5],
            strides: vec![0, 1],
            offset: 0,
        };
        let attempt_reshape = shape.attempt_reshape_without_copying(&[4, 2, 5]);
        assert_eq!(attempt_reshape.unwrap().strides, &[0, 0, 1]);

        // expanded at dim 0
        let shape = Shape {
            shape: vec![3, 6],
            strides: vec![0, 1],
            offset: 0,
        };
        let attempt_reshape = shape.attempt_reshape_without_copying(&[2, 3, 3]);
        assert!(attempt_reshape.is_err());

        // expanded at dim 0
        let shape = Shape {
            shape: vec![3, 6],
            strides: vec![0, 1],
            offset: 0,
        };
        let attempt_reshape = shape.attempt_reshape_without_copying(&[3, 2, 3]);
        assert_eq!(attempt_reshape.unwrap().strides, &[0, 3, 1]);

        // expanded at dim 0
        let shape = Shape {
            shape: vec![6],
            strides: vec![0],
            offset: 0,
        };
        let attempt_reshape = shape.attempt_reshape_without_copying(&[1, 1, 6]);
        assert_eq!(attempt_reshape.unwrap().strides, &[0, 0, 0]);

        // expanded at dim 1
        let shape = Shape {
            shape: vec![4, 3, 2],
            strides: vec![2, 0, 1],
            offset: 0,
        };
        let attempt_reshape = shape.attempt_reshape_without_copying(&[4, 3, 1, 2]);
        assert_eq!(attempt_reshape.unwrap().strides, &[2, 0, 2, 1]);

        // transpose or permute
        let shape = Shape {
            shape: vec![4, 3, 2],
            strides: vec![6, 2, 1],
            offset: 0,
        }
        .permute(&[2, 1, 0]);
        let attempt_reshape = shape.attempt_reshape_without_copying(&[4, 1, 3, 2]);
        assert!(attempt_reshape.is_err());
    }
}
