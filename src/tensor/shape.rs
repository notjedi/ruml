use crate::assert_dim;

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
}
