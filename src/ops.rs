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
    Log,
    Negate,
    Sqrt,
    Square,
    Sum,

    // BinaryOps
    Add,
    Div,
    Mul,
    Sub,

    // MlOps
    GeLU,
    MatMul,
    Mean,
    Norm,
    ReLU,
    Scale,
    SiLU,
    Sigmoid,
    Softmax,

    // ShapeOps
    Expand,
    Permute,
    Reshape,
    Transpose,
    View,
}
