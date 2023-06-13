pub mod cpu;

pub use cpu::CpuBackend;

use crate::{types::NumFloat, Tensor};

pub trait Backend<T>
where
    T: NumFloat,
{
    fn matmul();
    fn sum(tensor: &Tensor<T>) -> T;
    fn sum_axis(tensor: &Tensor<f32>, dim: usize) -> Tensor<T>;
    fn add();
}

// https://users.rust-lang.org/t/unit-tests-for-traits/86848
pub mod tests {
    use aligned_vec::avec;

    use super::*;

    pub fn test_matmul<T: Backend<f32>>() {
        let out = T::matmul();
        assert_eq!(out, ());
    }

    pub fn test_sum<T: Backend<f32>>() {
        let tensor = Tensor::ones(&[4, 3, 3]).contiguous();
        let out = T::sum(&tensor);
        assert_eq!(out, tensor.numel() as f32);
        let tensor = Tensor::new(avec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let out = T::sum(&tensor);
        assert_eq!(out, 55.0);
    }
}
