pub mod cpu;

pub use cpu::AVX2Backend;

use crate::{types::NumFloat, Tensor};

pub trait Backend<T>
where
    T: NumFloat,
{
    fn matmul();
    fn relu(tensor: &Tensor<T>) -> Tensor<T>;
    fn sum(tensor: &Tensor<T>) -> T;
    fn add_scalar(a: &Tensor<T>, b: T) -> Tensor<T>;
    fn add_elementwise(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;
}

// https://users.rust-lang.org/t/unit-tests-for-traits/86848
pub mod tests {
    use std::marker::PhantomData;

    use aligned_vec::{AVec, CACHELINE_ALIGN};

    use super::*;

    pub struct Tests<T, U>
    where
        T: Backend<U>,
        U: NumFloat,
    {
        backend: PhantomData<T>,
        dtype: PhantomData<U>,
    }

    impl<T, U> Tests<T, U>
    where
        T: Backend<U>,
        U: NumFloat,
    {
        pub fn test_matmul() {
            let out = T::matmul();
            assert_eq!(out, ());
        }

        pub fn test_relu() {
            let vals_iter = (-5..5).map(|x| U::from(x).unwrap());
            let a = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
            let out = T::relu(&a);
            assert_eq!(
                out.ravel(),
                AVec::from_iter(
                    CACHELINE_ALIGN,
                    [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
                        .into_iter()
                        .map(|x| U::from(x).unwrap())
                )
            );
        }

        pub fn test_sum() {
            let tensor = Tensor::ones(&[4, 3, 3]).contiguous();
            let out = T::sum(&tensor);
            assert_eq!(out, U::from(tensor.numel()).unwrap());

            let vals_iter = (1..11).map(|x| U::from(x).unwrap());
            let tensor = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
            let out = T::sum(&tensor);
            assert_eq!(out, U::from(55).unwrap());
        }

        pub fn test_add_elementwise() {
            let len = [4, 3, 3].iter().product();
            let a = Tensor::<U>::arange(len);
            let b = Tensor::<U>::arange(len);
            let out = T::add_elementwise(&a, &b);
            (0..len)
                .into_iter()
                .zip(out.into_iter())
                .enumerate()
                .for_each(|(i, (base, res))| {
                    assert_eq!(
                        U::from(base * 2).unwrap(),
                        res,
                        "results don't match at index {}",
                        i
                    )
                });
        }

        pub fn test_add_scalar() {
            let len = [4, 3, 3].iter().product();
            let a = Tensor::<U>::arange(len);
            let b = U::from(10).unwrap();
            let out = T::add_scalar(&a, b);
            (0..len)
                .into_iter()
                .zip(out.into_iter())
                .enumerate()
                .for_each(|(i, (base, res))| {
                    assert_eq!(
                        U::from(base + 10).unwrap(),
                        res,
                        "results don't match at index {}",
                        i
                    )
                });
        }
    }
}
