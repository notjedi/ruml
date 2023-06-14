pub mod cpu;

pub use cpu::AVX2Backend;

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
    use aligned_vec::{AVec, CACHELINE_ALIGN};

    use super::*;

    pub fn test_matmul<T, U>()
    where
        T: Backend<U>,
        U: NumFloat,
    {
        let out = T::matmul();
        assert_eq!(out, ());
    }

    pub fn test_sum<T, U>()
    where
        T: Backend<U>,
        U: NumFloat,
    {
        let tensor = Tensor::ones(&[4, 3, 3]).contiguous();
        let out = T::sum(&tensor);
        assert_eq!(out, U::from(tensor.numel()).unwrap());

        let vals_iter = (1..11).map(|x| U::from(x).unwrap());
        let tensor = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
        let out = T::sum(&tensor);
        assert_eq!(out, U::from(55).unwrap());
    }

    pub fn test_add_elementwise<T, U>()
    where
        T: Backend<U>,
        U: NumFloat,
    {
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

    pub fn test_add_scalar<T, U>()
    where
        T: Backend<U>,
        U: NumFloat,
    {
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
