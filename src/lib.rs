#![feature(portable_simd)]
#![feature(slice_as_chunks)]

mod backend;
mod ops;
mod tensor;
pub mod types;

pub use backend::*;
pub use ops::*;
pub use tensor::*;

macro_rules! assert_dim {
    ($dim:expr, $ndim:expr) => {
        assert!(
            $dim < $ndim,
            "{} should be within the range of 0 <= dim < {}",
            $dim,
            $ndim
        );
    };
    ($dim1:expr, $dim2:expr, $ndim:expr) => {
        assert!(
            $dim1 < $ndim && $dim2 < $ndim,
            "both dim1({}) and dim2({}) should be less than {}",
            $dim1,
            $dim2,
            $ndim
        );
    };
}

macro_rules! assert_numel {
    ($self_numel:expr, $other:ident) => {
        assert_eq!(
            $self_numel,
            $other.numel(),
            "shape {:?} is invalid for input of size {}.",
            $other,
            $self_numel
        );
    };
    ($self_numel:expr, $other_numel:expr, $other:ident) => {
        assert_eq!(
            $self_numel, $other_numel,
            "shape {:?} is invalid for input of size {}.",
            $other, $self_numel
        );
    };
}

pub(crate) use assert_dim;
pub(crate) use assert_numel;
