pub mod graph_tensor;

use core::marker::PhantomData;
use daggy::Dag;

use crate::{types::NumFloat, Backend, Op, Tensor};
use graph_tensor::GraphTensor;

pub struct Graph<T, U>
where
    T: Backend<U>,
    U: NumFloat,
{
    pub(crate) graph: Dag<Tensor<U>, Op>,
    phantom_data: PhantomData<T>,
}

impl<T, U> Graph<T, U>
where
    T: Backend<U>,
    U: NumFloat,
{
    pub fn arange(&mut self, len: usize) -> GraphTensor<T, U> {
        let tensor = Tensor::arange(len);
        let idx = self.graph.add_node(tensor);
        GraphTensor {
            id: idx,
            graph_ref: self,
        }
    }
}
