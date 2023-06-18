use aligned_vec::{avec, AVec};

use super::Backend;
use crate::{Tensor, CACHELINE_ALIGN};
use alloc::sync::Arc;
use core::ops::Add;
use core::simd::{f32x8, SimdFloat};
use std::simd::StdFloat;

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
        let (_, aligned, suffix) = tensor.data.as_simd::<8>();
        let zeros = f32x8::splat(0.0);

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

    fn sqrt(tensor: &Tensor<f32>) -> Tensor<f32> {
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());
        let (_, aligned, suffix) = tensor.data.as_simd::<8>();

        aligned.iter().for_each(|simd_chunk| {
            let sqrt = simd_chunk.sqrt();
            sqrt.as_array().iter().for_each(|&elem| data.push(elem));
        });
        suffix.iter().for_each(|elem| {
            data.push(elem.sqrt());
        });

        Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
        }
    }

    fn log2(tensor: &Tensor<f32>) -> Tensor<f32> {
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());
        tensor.data.array_chunks::<8>().for_each(|&chunk| {
            let log = wide::f32x8::from(chunk).log2();
            log.to_array().into_iter().for_each(|val| data.push(val));
        });
        tensor.data[(data.len() / 8) * 8..]
            .iter()
            .for_each(|val| data.push(val.log2()));

        Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
        }
    }

    fn exp(tensor: &Tensor<f32>) -> Tensor<f32> {
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());
        tensor.data.array_chunks::<8>().for_each(|&chunk| {
            let exp = wide::f32x8::from(chunk).exp();
            exp.to_array().into_iter().for_each(|val| data.push(val));
        });
        tensor.data[(data.len() / 8) * 8..]
            .iter()
            .for_each(|val| data.push(val.exp()));

        Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
        }
    }

    fn sigmoid(tensor: &Tensor<f32>) -> Tensor<f32> {
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());
        tensor
            .data
            .array_chunks::<{ Self::CHUNK_SIZE }>()
            .for_each(|&chunk| {
                let chunk_simd = wide::f32x8::from(chunk);
                let sigmoid = (1.0 + (-chunk_simd).exp()).recip();
                sigmoid
                    .to_array()
                    .into_iter()
                    .for_each(|val| data.push(val));
            });
        tensor.data[(data.len() / Self::CHUNK_SIZE) * Self::CHUNK_SIZE..]
            .iter()
            .for_each(|val| data.push(1.0 / (1.0 + (-val).exp())));

        Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
        }
    }

    fn silu(tensor: &Tensor<f32>) -> Tensor<f32> {
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());

        tensor
            .data
            .array_chunks::<{ Self::CHUNK_SIZE }>()
            .for_each(|&chunk| {
                let chunk_simd = wide::f32x8::from(chunk);
                let sigmoid = (1.0 + (-chunk_simd).exp()).recip();
                let silu = chunk_simd * sigmoid;
                silu.to_array().into_iter().for_each(|val| data.push(val));
            });
        tensor.data[(data.len() / Self::CHUNK_SIZE) * Self::CHUNK_SIZE..]
            .iter()
            .for_each(|val| {
                let sigmoid = 1.0 / (1.0 + (-val).exp());
                data.push(val * sigmoid);
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
        let (_, aligned, suffix) = tensor.data.as_simd::<8>();
        let acc = f32x8::splat(0.0);
        let acc = aligned.iter().fold(acc, f32x8::add);
        acc.reduce_sum() + suffix.iter().sum::<f32>()
    }

    fn sum_axis(tensor: &Tensor<f32>, dim: usize) -> Tensor<f32> {
        // i'm not really satisfied w this code, kinda feels messy and easy to break also it
        // involves a lot of copying data to simd registers, so i'm kinda skeptical about this one
        // and we'll have to see.
        debug_assert!(
            tensor.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        assert!(
            dim < tensor.shape.ndim(),
            "dim must be less than {}",
            tensor.shape.ndim()
        );
        let stride = tensor.shape.strides[0];
        let row_stride = tensor.shape.strides[dim];
        let new_shape = tensor.shape.remove_dim(dim);

        fn sum_over_last_dim(tensor: &Tensor<f32>, dim: usize) -> Tensor<f32> {
            let last_dim_elems = tensor.shape.shape[dim];
            let new_shape = tensor.shape.remove_dim(dim);
            let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());

            if last_dim_elems < 8 {
                tensor.data.chunks_exact(last_dim_elems).for_each(|chunk| {
                    data.push(chunk.iter().sum());
                });
            } else {
                tensor.data.chunks_exact(last_dim_elems).for_each(|chunk| {
                    let mut sum = 0.0;
                    chunk.array_chunks::<8>().for_each(|&oct_chunk| {
                        let aligned = f32x8::from_array(oct_chunk);
                        sum += aligned.reduce_sum();
                    });
                    sum += chunk[(chunk.len() / 8) * 8..].iter().sum::<f32>();
                    data.push(sum);
                });
            }
            Tensor {
                data: Arc::new(data),
                shape: new_shape,
            }
        }

        match dim {
            0 => {
                let mut data = avec![0.0 as f32; new_shape.numel()];
                let (_, d_aligned, d_suffix) = data.as_simd_mut::<8>();

                tensor.data.chunks_exact(stride).for_each(|row| {
                    row.array_chunks::<8>().zip(d_aligned.iter_mut()).for_each(
                        |(&chunk, d_simd)| {
                            let aligned = f32x8::from_array(chunk);
                            *d_simd += aligned;
                        },
                    );

                    row[(row.len() / 8) * 8..]
                        .iter()
                        .zip(d_suffix.iter_mut())
                        .for_each(|(&row_data, orig_data)| *orig_data += row_data);
                });
                Tensor {
                    data: Arc::new(data),
                    shape: new_shape,
                }
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
                            row.chunks_exact(row_stride).for_each(|chunk| {
                                chunk
                                    .array_chunks::<8>()
                                    .zip(d_aligned.iter_mut())
                                    .for_each(|(&oct_chunk, d_simd)| {
                                        let aligned = f32x8::from_array(oct_chunk);
                                        *d_simd += aligned;
                                    });
                                chunk[chunk.len() - d_suffix.len()..]
                                    .iter()
                                    .zip(d_suffix.iter_mut())
                                    .for_each(|(elem, suffix)| {
                                        *suffix += elem;
                                    });
                            });
                        }
                    }
                    acc.iter().for_each(|&val| data.push(val));
                });
                Tensor {
                    data: Arc::new(data),
                    shape: new_shape,
                }
            }
            2 => {
                let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());

                if row_stride == 1 {
                    // 3-D tensor
                    sum_over_last_dim(tensor, dim)
                } else {
                    // 4-D tensor
                    let prev_stride = tensor.shape.strides[dim - 1];
                    tensor.data.chunks_exact(prev_stride).for_each(|row| {
                        let mut acc = avec![0.0 as f32; row_stride];
                        let (_, d_aligned, d_suffix) = acc.as_simd_mut::<8>();

                        row.chunks_exact(row_stride).for_each(|chunk| {
                            chunk
                                .array_chunks::<8>()
                                .zip(d_aligned.iter_mut())
                                .for_each(|(&oct_chunk, d_simd)| {
                                    let aligned = f32x8::from_array(oct_chunk);
                                    *d_simd += aligned;
                                });
                            chunk[chunk.len() - d_suffix.len()..]
                                .iter()
                                .zip(d_suffix.iter_mut())
                                .for_each(|(&elem, suffix)| {
                                    *suffix += elem;
                                });
                        });
                        acc.iter().for_each(|&val| data.push(val));
                    });
                    Tensor {
                        data: Arc::new(data),
                        shape: new_shape,
                    }
                }
            }
            3 => sum_over_last_dim(tensor, dim),
            _ => panic!("you shouldn't be here bro"),
        }
    }

    fn add_scalar(a: &Tensor<f32>, b: f32) -> Tensor<f32> {
        let b_vec = f32x8::splat(b);
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, a.data.len());
        let (_, a_aligned, a_suffix) = a.data.as_simd::<8>();

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

    fn sub_scalar(a: &Tensor<f32>, b: f32) -> Tensor<f32> {
        let b_vec = f32x8::splat(b);
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, a.data.len());
        let (_, a_aligned, a_suffix) = a.data.as_simd::<8>();

        a_aligned.iter().for_each(|a_vec| {
            let add_vec = a_vec - b_vec;
            add_vec.as_array().iter().for_each(|&elem| data.push(elem));
        });
        a_suffix.iter().for_each(|a_elem| {
            data.push(a_elem - b);
        });

        Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
        }
    }

    fn mul_scalar(a: &Tensor<f32>, b: f32) -> Tensor<f32> {
        let b_vec = f32x8::splat(b);
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, a.data.len());
        let (_, a_aligned, a_suffix) = a.data.as_simd::<8>();

        a_aligned.iter().for_each(|a_vec| {
            let add_vec = a_vec * b_vec;
            add_vec.as_array().iter().for_each(|&elem| data.push(elem));
        });
        a_suffix.iter().for_each(|a_elem| {
            data.push(a_elem * b);
        });

        Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
        }
    }

    fn div_scalar(a: &Tensor<f32>, b: f32) -> Tensor<f32> {
        let b_vec = f32x8::splat(b);
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, a.data.len());
        let (_, a_aligned, a_suffix) = a.data.as_simd::<8>();

        a_aligned.iter().for_each(|a_vec| {
            let add_vec = a_vec / b_vec;
            add_vec.as_array().iter().for_each(|&elem| data.push(elem));
        });
        a_suffix.iter().for_each(|a_elem| {
            data.push(a_elem / b);
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
        let (_, a_aligned, a_suffix) = a.data.as_simd::<8>();
        let (_, b_aligned, b_suffix) = b.data.as_simd::<8>();

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

    fn sub_elementwise(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
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
        let (_, a_aligned, a_suffix) = a.data.as_simd::<8>();
        let (_, b_aligned, b_suffix) = b.data.as_simd::<8>();

        a_aligned.iter().zip(b_aligned).for_each(|(a_vec, b_vec)| {
            let add_vec = a_vec - b_vec;
            add_vec.as_array().iter().for_each(|&elem| data.push(elem));
        });
        a_suffix.iter().zip(b_suffix).for_each(|(a_elem, b_elem)| {
            data.push(a_elem - b_elem);
        });

        Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
        }
    }

    fn mul_elementwise(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
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
        let (_, a_aligned, a_suffix) = a.data.as_simd::<8>();
        let (_, b_aligned, b_suffix) = b.data.as_simd::<8>();

        a_aligned.iter().zip(b_aligned).for_each(|(a_vec, b_vec)| {
            let add_vec = a_vec * b_vec;
            add_vec.as_array().iter().for_each(|&elem| data.push(elem));
        });
        a_suffix.iter().zip(b_suffix).for_each(|(a_elem, b_elem)| {
            data.push(a_elem * b_elem);
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
    fn test_log2() {
        backend_tests::<AVX2Backend, f32>::test_log2();
    }

    #[test]
    fn test_exp() {
        backend_tests::<AVX2Backend, f32>::test_exp();
    }

    #[test]
    fn test_silu() {
        backend_tests::<AVX2Backend, f32>::test_silu();
    }

    #[test]
    fn test_sigmoid() {
        backend_tests::<AVX2Backend, f32>::test_sigmoid();
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
