use std::fmt::Debug;

mod backend;
mod tensor;

pub use backend::*;
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
}

pub(crate) use assert_dim;
pub(crate) use assert_numel;

// List of ops taken from:
// 1. https://github.com/geohot/tinygrad/blob/master/tinygrad/ops.py
// 2. https://github.com/geohot/tinygrad/blob/master/tinygrad/mlops.py
// 3. https://github.com/ggerganov/ggml/blob/master/include/ggml/ggml.h#L236
// 4. https://github.com/apple/coremltools/blob/main/mlmodel/build/format/NeuralNetwork.pb.h#L2838
// 5. https://github.com/apple/coremltools/blob/main/mlmodel/src/Validation/NeuralNetwork/NeuralNetworkValidator.cpp#L38
pub enum UnaryOps {
    Abs,
    Copy,
    Negate,
    None,
    Square,
    SquareRoot,
    // Exp, // ?
}

pub enum BinaryOps {
    Add,
    Div,
    Mul,
    Sub,
}

pub enum MlOps {
    Dot,
    GeLU,
    Mean,
    Norm,
    ReLU,
    Scale,
    SiLU,
    Softmax,
    Sum,
}

pub enum ShapeOps {
    Permute,
    Repeat, // TODO: check if expand is actually repeat
    Reshape,
    Transpose,
    View,
}

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
