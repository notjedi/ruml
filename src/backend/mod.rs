pub mod cpu;

pub use cpu::AVX2Backend;

use crate::{types::NumFloat, Tensor};

pub trait Backend<T>
where
    T: NumFloat,
{
    const CHUNK_SIZE: usize;

    fn matmul(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;
    fn matmul_naive(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;

    fn exp(tensor: &Tensor<T>) -> Tensor<T>;
    fn log2(tensor: &Tensor<T>) -> Tensor<T>;
    fn relu(tensor: &Tensor<T>) -> Tensor<T>;
    fn sqrt(tensor: &Tensor<T>) -> Tensor<T>;
    fn silu(tensor: &Tensor<T>) -> Tensor<T>;
    fn sigmoid(tensor: &Tensor<T>) -> Tensor<T>;

    fn sum(tensor: &Tensor<T>) -> T;
    fn sum_axis(tensor: &Tensor<T>, dim: usize) -> Tensor<T>;

    // TODO: use macros to impl these methods
    fn add_scalar(a: &Tensor<T>, b: T) -> Tensor<T>;
    fn sub_scalar(a: &Tensor<T>, b: T) -> Tensor<T>;
    fn mul_scalar(a: &Tensor<T>, b: T) -> Tensor<T>;
    fn div_scalar(a: &Tensor<T>, b: T) -> Tensor<T>;

    fn add_elementwise(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;
    fn sub_elementwise(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;
    fn mul_elementwise(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;
}

// https://users.rust-lang.org/t/unit-tests-for-traits/86848
#[path = "./backend_test.rs"]
pub mod tests;
