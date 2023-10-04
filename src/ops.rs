// List of ops taken from:
// 1. https://github.com/geohot/tinygrad/blob/master/tinygrad/ops.py
// 2. https://github.com/geohot/tinygrad/blob/master/tinygrad/mlops.py
// 3. https://github.com/ggerganov/ggml/blob/master/include/ggml/ggml.h#L236
// 4. https://github.com/apple/coremltools/blob/main/mlmodel/build/format/NeuralNetwork.pb.h#L2838
// 5. https://github.com/apple/coremltools/blob/main/mlmodel/src/Validation/NeuralNetwork/NeuralNetworkValidator.cpp#L38

use core::fmt::Debug;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Op {
    Noop,
    Todo,

    // UnaryOps
    Abs,
    Copy,
    Negate,
    None,
    Square,
    SquareRoot,

    // BinaryOps
    Add,
    Div,
    Mul,
    Sub,

    // MlOps
    Dot,
    GeLU,
    Mean,
    Norm,
    ReLU,
    Scale,
    SiLU,
    Softmax,
    Sum,

    // ShapeOps
    Expand,
    Permute,
    Reshape,
    Transpose,
    View,
}

impl Default for Op {
    fn default() -> Self {
        Op::Noop
    }
}

impl Debug for Op {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Noop => write!(f, "Noop"),
            Self::Todo => write!(f, "Todo"),
            Self::Abs => write!(f, "Abs"),
            Self::Copy => write!(f, "Copy"),
            Self::Negate => write!(f, "Negate"),
            Self::None => write!(f, "None"),
            Self::Square => write!(f, "Square"),
            Self::SquareRoot => write!(f, "SquareRoot"),
            Self::Add => write!(f, "Add"),
            Self::Div => write!(f, "Div"),
            Self::Mul => write!(f, "Mul"),
            Self::Sub => write!(f, "Sub"),
            Self::Dot => write!(f, "Dot"),
            Self::GeLU => write!(f, "GeLU"),
            Self::Mean => write!(f, "Mean"),
            Self::Norm => write!(f, "Norm"),
            Self::ReLU => write!(f, "ReLU"),
            Self::Scale => write!(f, "Scale"),
            Self::SiLU => write!(f, "SiLU"),
            Self::Softmax => write!(f, "Softmax"),
            Self::Sum => write!(f, "Sum"),
            Self::Expand => write!(f, "Expand"),
            Self::Permute => write!(f, "Permute"),
            Self::Reshape => write!(f, "Reshape"),
            Self::Transpose => write!(f, "Transpose"),
            Self::View => write!(f, "View"),
        }
    }
}
