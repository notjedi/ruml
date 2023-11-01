// List of ops taken from:
// 1. https://github.com/geohot/tinygrad/blob/master/tinygrad/ops.py
// 2. https://github.com/geohot/tinygrad/blob/master/tinygrad/mlops.py
// 3. https://github.com/ggerganov/ggml/blob/master/include/ggml/ggml.h#L236
// 4. https://github.com/apple/coremltools/blob/main/mlmodel/build/format/NeuralNetwork.pb.h#L2838
// 5. https://github.com/apple/coremltools/blob/main/mlmodel/src/Validation/NeuralNetwork/NeuralNetworkValidator.cpp#L38

use core::fmt::Debug;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Op {
    #[default]
    Noop,

    // UnaryOps
    Abs,
    Exp,
    GeLU,
    Log,
    Negate,
    ReLU,
    SiLU,
    Sigmoid,
    Sqrt,
    Square,

    // UnaryOps with dim
    Mean,
    Norm,
    Softmax,
    Sum,

    // BinaryOps
    Add,
    Div,
    Mul,
    Sub,

    // MlOps
    MatMul,

    // ShapeOps
    Expand,
    Permute,
    Reshape,
    Transpose,
    View,
}
