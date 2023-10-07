mod graph_tensor;
pub use graph_tensor::GraphTensor;

use core::marker::PhantomData;
use daggy::Dag;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::{types::NumFloat, Backend, Op, Tensor};

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

    pub fn eye(&mut self, dim: usize) -> GraphTensor<T, U> {
        let tensor = Tensor::eye(dim);
        let idx = self.graph.add_node(tensor);
        GraphTensor {
            id: idx,
            graph_ref: self,
        }
    }

    pub fn tril(&mut self, dim: usize) -> GraphTensor<T, U> {
        let tensor = Tensor::tril(dim);
        let idx = self.graph.add_node(tensor);
        GraphTensor {
            id: idx,
            graph_ref: self,
        }
    }

    pub fn triu(&mut self, dim: usize) -> GraphTensor<T, U> {
        let tensor = Tensor::triu(dim);
        let idx = self.graph.add_node(tensor);
        GraphTensor {
            id: idx,
            graph_ref: self,
        }
    }

    pub fn randn<R>(&mut self, shape: &[usize], rng: &mut R) -> GraphTensor<T, U>
    where
        R: Rng,
        StandardNormal: Distribution<U>,
    {
        let tensor = Tensor::randn(shape, rng);
        let idx = self.graph.add_node(tensor);
        GraphTensor {
            id: idx,
            graph_ref: self,
        }
    }

    pub fn full(&mut self, fill_value: U, shape: &[usize]) -> GraphTensor<T, U> {
        let tensor = Tensor::full(fill_value, shape);
        let idx = self.graph.add_node(tensor);
        GraphTensor {
            id: idx,
            graph_ref: self,
        }
    }

    pub fn full_like(&mut self, fill_value: U, other: &GraphTensor<T, U>) -> GraphTensor<T, U> {
        let other_node = other.get_node_ref();
        let tensor = Tensor::full_like(fill_value, other_node);
        let idx = self.graph.add_node(tensor);
        GraphTensor {
            id: idx,
            graph_ref: self,
        }
    }

    pub fn ones(&mut self, shape: &[usize]) -> GraphTensor<T, U> {
        let tensor = Tensor::ones(shape);
        let idx = self.graph.add_node(tensor);
        GraphTensor {
            id: idx,
            graph_ref: self,
        }
    }

    pub fn zeros(&mut self, shape: &[usize]) -> GraphTensor<T, U> {
        let tensor = Tensor::zeros(shape);
        let idx = self.graph.add_node(tensor);
        GraphTensor {
            id: idx,
            graph_ref: self,
        }
    }

    pub fn ones_like(&mut self, other: &GraphTensor<T, U>) -> GraphTensor<T, U> {
        let other_node = other.get_node_ref();
        let tensor = Tensor::ones_like(other_node);
        let idx = self.graph.add_node(tensor);
        GraphTensor {
            id: idx,
            graph_ref: self,
        }
    }

    pub fn zeros_like(&mut self, other: &GraphTensor<T, U>) -> GraphTensor<T, U> {
        let other_node = other.get_node_ref();
        let tensor = Tensor::zeros_like(other_node);
        let idx = self.graph.add_node(tensor);
        GraphTensor {
            id: idx,
            graph_ref: self,
        }
    }
}
