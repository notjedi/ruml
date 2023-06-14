use aligned_vec::{AVec, CACHELINE_ALIGN};

use super::Backend;
use crate::{assert_prefix_len, Tensor};
use core::simd::{f32x8, SimdFloat};
use std::{ops::Add, sync::Arc};

pub struct AVX2Backend {}

impl Backend<f32> for AVX2Backend {
    fn matmul() {
        todo!()
    }

    fn sum(tensor: &Tensor<f32>) -> f32 {
        debug_assert!(
            tensor.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        let (prefix, aligned, suffix) = tensor.data.as_simd::<8>();
        assert_prefix_len!(prefix);
        let acc = f32x8::splat(0.0);
        let acc = aligned.iter().fold(acc, f32x8::add);
        acc.reduce_sum() + suffix.iter().sum::<f32>()
    }

    fn add_elementwise(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        assert!(a.shape() == b.shape(), "len of both tensors should match");
        debug_assert!(
            a.is_contiguous() && b.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        debug_assert!(
            a.data.alignment() == b.data.alignment(),
            "data must be aligned"
        );

        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, a.data.len());
        let (a_prefix, a_aligned, a_suffix) = a.data.as_simd::<8>();
        let (b_prefix, b_aligned, b_suffix) = b.data.as_simd::<8>();
        assert_prefix_len!(a_prefix);
        assert_prefix_len!(b_prefix);

        a_aligned.iter().zip(b_aligned).for_each(|(a_vec, b_vec)| {
            let add_vec = a_vec + b_vec;
            add_vec.as_array().iter().for_each(|&elem| data.push(elem));
        });
        a_suffix.iter().zip(b_suffix).for_each(|(a_elem, b_elem)| {
            data.push(a_elem + b_elem);
        });

        Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
        }
    }

    fn add_scalar(a: &Tensor<f32>, b: f32) -> Tensor<f32> {
        let b_vec = f32x8::splat(b);
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, a.data.len());
        let (a_prefix, a_aligned, a_suffix) = a.data.as_simd::<8>();
        assert_prefix_len!(a_prefix);

        a_aligned.iter().for_each(|a_vec| {
            let add_vec = a_vec + b_vec;
            add_vec.as_array().iter().for_each(|&elem| data.push(elem));
        });
        a_suffix.iter().for_each(|a_elem| {
            data.push(a_elem + b);
        });

        Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AVX2Backend;
    use crate::backend::tests as backend_tests;

    #[test]
    #[ignore = "unimplemented"]
    fn test_matmul() {
        backend_tests::test_matmul::<AVX2Backend, f32>();
    }

    #[test]
    fn test_sum() {
        backend_tests::test_sum::<AVX2Backend, f32>();
    }

    #[test]
    fn test_add_scalar() {
        backend_tests::test_add_scalar::<AVX2Backend, f32>();
    }

    #[test]
    fn test_add_elementwise() {
        backend_tests::test_add_elementwise::<AVX2Backend, f32>();
    }
}
