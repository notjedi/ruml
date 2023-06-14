pub mod cpu;

pub use cpu::CpuBackend;

use crate::{types::NumFloat, Tensor};

pub trait Backend<T>
where
    T: NumFloat,
{
    fn matmul();
    fn sum(tensor: &Tensor<T>) -> T;
    fn add_elementwise(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;
    fn add_scalar(a: &Tensor<T>, b: T) -> Tensor<T>;
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

    pub fn test_add_elementwise<T: Backend<f32>>() {
        let len = [4, 3, 3].iter().product();
        let a = Tensor::<f32>::arange(len);
        let b = Tensor::<f32>::arange(len);
        let out = T::add_elementwise(&a, &b);
        (0..len)
            .into_iter()
            .zip(out.into_iter())
            .enumerate()
            .for_each(|(i, (base, res))| {
                assert_eq!((base * 2) as f32, res, "results don't match at index {}", i)
            });
    }

    pub fn test_add_scalar<T: Backend<f32>>() {
        let len = [4, 3, 3].iter().product();
        let a = Tensor::<f32>::arange(len);
        let b: f32 = 10.0;
        let out = T::add_scalar(&a, b);
        (0..len)
            .into_iter()
            .zip(out.into_iter())
            .enumerate()
            .for_each(|(i, (base, res))| {
                assert_eq!(
                    base as f32 + 10.0,
                    res,
                    "results don't match at index {}",
                    i
                )
            });
    }
}
