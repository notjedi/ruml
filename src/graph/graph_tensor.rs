use daggy::{EdgeIndex, NodeIndex};

use crate::{graph::Graph, types::NumFloat, Backend, Op, Tensor};

pub struct GraphTensor<T, U>
where
    T: Backend<U>,
    U: NumFloat,
{
    pub(crate) id: NodeIndex,
    pub(crate) graph_ref: *mut Graph<T, U>,
}

// TODO: impl unary ops
impl<T, U> GraphTensor<T, U>
where
    T: Backend<U>,
    U: NumFloat,
{
    fn add_child(&self, child: Tensor<U>, op: Op) -> (EdgeIndex, NodeIndex) {
        unsafe { (*self.graph_ref).graph.add_child(self.id, op, child) }
    }

    pub(crate) fn get_node_ref(&self) -> &Tensor<U> {
        unsafe { &(*self.graph_ref).graph[self.id] }
    }

    pub fn abs(&self) -> GraphTensor<T, U> {
        let node = self.get_node_ref();
        let abs_tensor = T::abs(node);
        let (_, idx) = self.add_child(abs_tensor, Op::Abs);
        GraphTensor {
            id: idx,
            graph_ref: self.graph_ref,
        }
    }

    pub fn exp(&self) -> GraphTensor<T, U> {
        let node = self.get_node_ref();
        let exp_tensor = T::exp(node);
        let (_, idx) = self.add_child(exp_tensor, Op::Abs);
        GraphTensor {
            id: idx,
            graph_ref: self.graph_ref,
        }
    }

    pub fn log(&self) -> GraphTensor<T, U> {
        let node = self.get_node_ref();
        let log_tensor = T::log2(node);
        let (_, idx) = self.add_child(log_tensor, Op::Log);
        GraphTensor {
            id: idx,
            graph_ref: self.graph_ref,
        }
    }

    pub fn neg(&self) -> GraphTensor<T, U> {
        let node = self.get_node_ref();
        // TODO: should i use simd here?
        let neg_tensor = -node;
        let (_, idx) = self.add_child(neg_tensor, Op::Negate);
        GraphTensor {
            id: idx,
            graph_ref: self.graph_ref,
        }
    }

    pub fn sqrt(&self) -> GraphTensor<T, U> {
        let node = self.get_node_ref();
        let sqrt_tensor = T::sqrt(node);
        let (_, idx) = self.add_child(sqrt_tensor, Op::Sqrt);
        GraphTensor {
            id: idx,
            graph_ref: self.graph_ref,
        }
    }

    pub fn square(&self) -> GraphTensor<T, U> {
        let node = self.get_node_ref();
        let square_tensor = T::square(node);
        let (_, idx) = self.add_child(square_tensor, Op::Square);
        GraphTensor {
            id: idx,
            graph_ref: self.graph_ref,
        }
    }

    pub fn matmul(&self, other: GraphTensor<T, U>) -> GraphTensor<T, U> {
        let node = self.get_node_ref();
        let other_node = other.get_node_ref();
        let matmul_tensor = T::matmul(node, other_node);
        let (_, idx) = self.add_child(matmul_tensor, Op::MatMul);
        GraphTensor {
            id: idx,
            graph_ref: self.graph_ref,
        }
    }

    pub fn relu(&self) -> GraphTensor<T, U> {
        let node = self.get_node_ref();
        let relu_tensor = T::relu(node);
        let (_, idx) = self.add_child(relu_tensor, Op::ReLU);
        GraphTensor {
            id: idx,
            graph_ref: self.graph_ref,
        }
    }

    pub fn silu(&self) -> GraphTensor<T, U> {
        let node = self.get_node_ref();
        let silu_tensor = T::silu(node);
        let (_, idx) = self.add_child(silu_tensor, Op::SiLU);
        GraphTensor {
            id: idx,
            graph_ref: self.graph_ref,
        }
    }

    pub fn sigmoid(&self) -> GraphTensor<T, U> {
        let node = self.get_node_ref();
        let sigmoid_tensor = T::sigmoid(node);
        let (_, idx) = self.add_child(sigmoid_tensor, Op::Sigmoid);
        GraphTensor {
            id: idx,
            graph_ref: self.graph_ref,
        }
    }
}
