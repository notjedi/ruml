// List of ops taken from:
// 1. https://github.com/geohot/tinygrad/blob/master/tinygrad/ops.py
// 2. https://github.com/geohot/tinygrad/blob/master/tinygrad/mlops.py
// 3. https://github.com/ggerganov/ggml/blob/master/include/ggml/ggml.h#L236
// 4. https://github.com/apple/coremltools/blob/main/mlmodel/build/format/NeuralNetwork.pb.h#L2838
// 5. https://github.com/apple/coremltools/blob/main/mlmodel/src/Validation/NeuralNetwork/NeuralNetworkValidator.cpp#L38

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
