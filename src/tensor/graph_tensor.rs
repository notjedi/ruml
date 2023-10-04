use daggy::{EdgeIndex, NodeIndex};

use super::graph::Graph;
use crate::{types::NumFloat, Backend, Op, Tensor};

pub struct GraphTensor<T, U>
where
    T: Backend<U>,
    U: NumFloat,
{
    pub(crate) id: NodeIndex,
    pub(crate) graph_ref: *mut Graph<T, U>,
}

impl<T, U> GraphTensor<T, U>
where
    T: Backend<U>,
    U: NumFloat,
{
    fn get_node_ref(&self) -> &Tensor<U> {
        unsafe { &(*self.graph_ref).graph[self.id] }
    }

    fn add_child(&self, child: Tensor<U>, op: Op) -> (EdgeIndex, NodeIndex) {
        unsafe { (*self.graph_ref).graph.add_child(self.id, op, child) }
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
}
