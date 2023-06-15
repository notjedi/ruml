use aligned_vec::{avec, AVec, CACHELINE_ALIGN};

use super::Backend;
use crate::{assert_prefix_len, Tensor};
use alloc::sync::Arc;
use core::ops::Add;
use core::simd::{f32x8, SimdFloat};

pub struct AVX2Backend;

impl Backend<f32> for AVX2Backend {
    fn matmul() {
        todo!()
    }

    fn relu(tensor: &Tensor<f32>) -> Tensor<f32> {
        debug_assert!(
            tensor.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());
        let (prefix, aligned, suffix) = tensor.data.as_simd::<8>();
        let zeros = f32x8::splat(0.0);
        assert_prefix_len!(prefix);

        aligned.iter().for_each(|&vec| {
            let mask = vec.is_sign_positive();
            let masked_elems = mask.select(vec, zeros);
            masked_elems
                .as_array()
                .iter()
                .for_each(|&elem| data.push(elem));
        });
        suffix.iter().for_each(|&x| {
            if x < 0.0 {
                data.push(0.0)
            } else {
                data.push(x)
            }
        });

        Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
        }
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

    // fn sum_axis(tensor: &Tensor<f32>, dim: usize) -> Tensor<f32> {
    fn sum_axis(tensor: &Tensor<f32>, dim: usize) {
        debug_assert!(
            tensor.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        let stride = tensor.shape.strides[0];
        let row_stride = tensor.shape.strides[dim];
        let new_shape = tensor.shape.remove_dim(dim);

        match dim {
            0 => {
                let mut data = avec![0.0 as f32; new_shape.numel()];
                let (_, d_aligned, d_suffix) = data.as_simd_mut::<8>();

                tensor.data.chunks_exact(stride).for_each(|row| {
                    row.chunks_exact(8)
                        .zip(d_aligned.iter_mut())
                        .for_each(|(chunk, d_simd)| {
                            let aligned = f32x8::from_slice(chunk);
                            *d_simd += aligned;
                        });

                    row[(row.len() / 8) * 8..]
                        .iter()
                        .zip(d_suffix.iter_mut())
                        .for_each(|(&row_data, orig_data)| *orig_data += row_data);
                });
                dbg!(&data);
                dbg!(&new_shape.shape());
            }
            1 => {
                let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());

                // TODO: using SIMD for 2-D tensors will only add to overhead, cause row_stride would be 1, so will be `stride` chunks
                tensor.data.chunks_exact(stride).for_each(|row| {
                    let mut acc = avec![0.0 as f32; row_stride];
                    {
                        let (_, d_aligned, d_suffix) = acc.as_simd_mut::<8>();
                        if d_aligned.len() == 0 {
                            // NOTE: row_stride < 8, so we need to iterate over the suffix bug, ig
                            // we are not guaranteed that d_aligned will have elements even if
                            // row_stride is > 8 and aligned. from:
                            // https://doc.rust-lang.org/std/primitive.slice.html#method.as_simd
                            row.chunks_exact(row_stride).for_each(|chunk| {
                                chunk
                                    .iter()
                                    .zip(d_suffix.iter_mut())
                                    .for_each(|(chunk, d_simd)| {
                                        *d_simd += chunk;
                                    })
                            });
                        } else {
                            row.chunks_exact(row_stride)
                                .zip(d_aligned.iter_mut())
                                .for_each(|(chunk, d_simd)| {
                                    let aligned = f32x8::from_slice(chunk);
                                    *d_simd += aligned;
                                });
                        }
                    }
                    acc.iter().for_each(|&val| data.push(val));
                });
                dbg!(&data);
                dbg!(&new_shape.shape());
            }
            2 => {
                let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());
                let row_stride = tensor.shape.strides[1];

                // TODO: using SIMD for 3-D tensors will only add to overhead, cause row_stride would be 1, so will be `stride` chunks
                tensor.data.chunks_exact(stride).for_each(|row| {
                    row.chunks_exact(row_stride).for_each(|chunk| {
                        if chunk.len() < 8 {
                            data.push(chunk.iter().sum());
                        } else {
                            let aligned = f32x8::from_slice(chunk);
                            data.push(aligned.reduce_sum());
                        }
                    });
                });
                dbg!(&data);
                dbg!(&new_shape.shape());
            }
            _ => {
                todo!()
            }
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
}

#[cfg(test)]
mod tests {
    use super::AVX2Backend;
    use crate::backend::tests::tests as backend_tests;

    #[test]
    #[ignore = "unimplemented"]
    fn test_matmul() {
        backend_tests::<AVX2Backend, f32>::test_matmul();
    }

    #[test]
    fn test_relu() {
        backend_tests::<AVX2Backend, f32>::test_relu();
    }

    #[test]
    fn test_sum() {
        backend_tests::<AVX2Backend, f32>::test_sum();
    }

    #[test]
    fn test_sum_axis() {
        backend_tests::<AVX2Backend, f32>::test_sum_axis();
    }

    #[test]
    fn test_add_scalar() {
        backend_tests::<AVX2Backend, f32>::test_add_scalar();
    }

    #[test]
    fn test_add_elementwise() {
        backend_tests::<AVX2Backend, f32>::test_add_elementwise();
    }
}
