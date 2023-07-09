use aligned_vec::{avec, AVec};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use super::Backend;
use crate::{Shape, Tensor, CACHELINE_ALIGN};
use alloc::sync::Arc;
use core::ops::Add;
use core::simd::{f32x8, SimdFloat};
use std::simd::StdFloat;

const CACHE_LINE_F32: usize = 16;

pub struct AVX2Backend;

#[repr(align(32))]
struct AlignedArray([f32; 16]);

impl Backend<f32> for AVX2Backend {
    const CHUNK_SIZE: usize = 8;

    fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        // TODO: only supports 2d tensors as of now

        let a_row = a.shape()[0];
        let b_row = b.shape()[0];
        let a_col = a.shape()[1];
        let b_col = b.shape()[1];
        assert_eq!(a_col, b_row);

        let new_shape = Shape::new(&[a_row, b_col]);
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());
        (0..new_shape.numel()).for_each(|_| data.push(0.0));

        for i in 0..a_row {
            let rem = b_col % CACHE_LINE_F32;
            let num_iters = b_col / CACHE_LINE_F32;
            let j_final = num_iters * CACHE_LINE_F32;
            let row = &a.data[i * a_col..i * a_col + a_col];

            for j in (0..num_iters).map(|x| x * CACHE_LINE_F32) {
                let mut buffer = AlignedArray([0.0 as f32; CACHE_LINE_F32]);

                row.iter().enumerate().for_each(|(k, &elem)| {
                    let elem_simd = f32x8::splat(elem);
                    let col = &b.data[(k * b_col) + j..(k * b_col) + j + CACHE_LINE_F32];

                    let (_, aligned, _) = buffer.0.as_simd_mut::<{ Self::CHUNK_SIZE }>();
                    aligned.iter_mut().zip(col.array_chunks::<8>()).for_each(
                        |(buf_mut, &col_chunk)| {
                            *buf_mut = elem_simd.mul_add(f32x8::from_array(col_chunk), *buf_mut)
                        },
                    );
                });

                data[(i * b_col) + j..(i * b_col) + j + CACHE_LINE_F32]
                    .copy_from_slice(buffer.0.as_slice());
            }

            let mut buffer = [0.0 as f32; CACHE_LINE_F32];
            row.iter().enumerate().for_each(|(k, &elem)| {
                let col = &b.data[(k * b_col) + j_final..(k * b_col) + j_final + rem];
                buffer
                    .iter_mut()
                    .zip(col.iter())
                    .for_each(|(dst, col_elem)| *dst += col_elem * elem);
            });

            data[(i * b_col) + j_final..(i * b_col) + j_final + rem]
                .iter_mut()
                .zip(&buffer[..rem])
                .for_each(|(dst, &src)| *dst += src);
        }

        Tensor {
            data: Arc::new(data),
            shape: new_shape,
        }
    }

    fn matmul_block(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        // NOTE:
        // torch is able to matmul two [2048, 2048] in 1.7 secs.
        // our impl takes about 4 secs.
        // we should at least get to 2 secs.
        // TODO: find what's causing the bottleneck
        let a_row = a.shape()[0];
        let b_row = b.shape()[0];
        let a_col = a.shape()[1];
        let b_col = b.shape()[1];
        assert_eq!(a_col, b_row);

        let new_shape = Shape::new(&[a_row, b_col]);
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());
        (0..new_shape.numel()).for_each(|_| data.push(0.0));

        let row_rem = a_row % CACHE_LINE_F32;
        let row_num_iters = a_row / CACHE_LINE_F32;
        let i_final = row_num_iters * CACHE_LINE_F32;

        // loop to split rows of a into 16x chunks
        for i_outer in (0..row_num_iters).map(|x| x * CACHE_LINE_F32) {
            let col_rem = b_col % CACHE_LINE_F32;
            let col_num_iters = b_col / CACHE_LINE_F32;
            let j_final = col_num_iters * CACHE_LINE_F32;

            // loop to split cols of b into 16x chunks
            for j_outer in (0..col_num_iters).map(|x| x * CACHE_LINE_F32) {
                // loop thro each of the 16 rows
                for i in i_outer..i_outer + CACHE_LINE_F32 {
                    let row = &a.data[i * a_col..i * a_col + a_col];

                    let line_rem = a_col % CACHE_LINE_F32;
                    let line_num_iters = a_col / CACHE_LINE_F32;

                    // loop 16x chunks of each row
                    for line_i in (0..line_num_iters).map(|x| x * CACHE_LINE_F32) {
                        let line = &row[line_i..line_i + CACHE_LINE_F32];
                        let mut buffer = AlignedArray([0.0 as f32; CACHE_LINE_F32]);

                        // loop thro each elem of 16x chunk of row
                        line.iter().enumerate().for_each(|(k, &elem)| {
                            let elem_simd = f32x8::splat(elem);
                            let col = &b.data[((line_i + k) * b_col) + j_outer
                                ..((line_i + k) * b_col) + j_outer + CACHE_LINE_F32];

                            let (_, aligned, _) = buffer.0.as_simd_mut::<{ Self::CHUNK_SIZE }>();
                            aligned
                                .iter_mut()
                                .zip(col.array_chunks::<{ Self::CHUNK_SIZE }>())
                                .for_each(|(buf_mut, &col_chunk)| {
                                    *buf_mut =
                                        elem_simd.mul_add(f32x8::from_array(col_chunk), *buf_mut)
                                });
                        });
                        data[(i * b_col) + j_outer..(i * b_col) + j_outer + CACHE_LINE_F32]
                            .iter_mut()
                            .zip(buffer.0.as_slice())
                            .for_each(|(dst, &src)| *dst += src);
                    }

                    // TODO: take care of line_rem here
                }
            }
        }

        Tensor {
            data: Arc::new(data),
            shape: new_shape,
        }
    }

    fn matmul_naive(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        // TODO: only supports 2d tensors as of now

        // NOTE: use rsplit to get the last x items?
        let a_row = a.shape()[0];
        let b_row = b.shape()[0];
        let a_col = a.shape()[1];
        let b_col = b.shape()[1];
        assert_eq!(a_col, b_row);

        let new_shape = Shape::new(&[a_row, b_col]);
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());
        (0..new_shape.numel()).for_each(|_| data.push(0.0));

        for i in 0..a_row {
            for k in 0..b_row {
                for j in 0..b_col {
                    data[i * b_col + j] += a.data[i * a_col + k] * b.data[k * b_col + j];
                }
            }
        }

        Tensor {
            data: Arc::new(data),
            shape: new_shape,
        }
    }

    fn exp(tensor: &Tensor<f32>) -> Tensor<f32> {
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());
        tensor
            .data
            .array_chunks::<{ Self::CHUNK_SIZE }>()
            .for_each(|&chunk| {
                let exp = wide::f32x8::from(chunk).exp();
                exp.to_array().into_iter().for_each(|val| data.push(val));
            });
        tensor.data[(data.len() / Self::CHUNK_SIZE) * Self::CHUNK_SIZE..]
            .iter()
            .for_each(|val| data.push(val.exp()));

        Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
        }
    }

    fn log2(tensor: &Tensor<f32>) -> Tensor<f32> {
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());
        tensor
            .data
            .array_chunks::<{ Self::CHUNK_SIZE }>()
            .for_each(|&chunk| {
                let log = wide::f32x8::from(chunk).log2();
                log.to_array().into_iter().for_each(|val| data.push(val));
            });
        tensor.data[(data.len() / Self::CHUNK_SIZE) * Self::CHUNK_SIZE..]
            .iter()
            .for_each(|val| data.push(val.log2()));

        Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
        }
    }

    fn relu(tensor: &Tensor<f32>) -> Tensor<f32> {
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());
        let (_, aligned, suffix) = tensor.data.as_simd::<{ Self::CHUNK_SIZE }>();
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
        let (_, aligned, suffix) = tensor.data.as_simd::<{ Self::CHUNK_SIZE }>();

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

    fn sum(tensor: &Tensor<f32>) -> f32 {
        debug_assert!(
            tensor.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        let (_, aligned, suffix) = tensor.data.as_simd::<{ Self::CHUNK_SIZE }>();
        let acc = f32x8::splat(0.0);
        let acc = aligned.iter().fold(acc, f32x8::add);
        acc.reduce_sum() + suffix.iter().sum::<f32>()
    }

    fn sum_rayon(tensor: &Tensor<f32>) -> f32 {
        debug_assert!(
            tensor.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        tensor.data.par_iter().sum::<f32>()
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
            // https://stackoverflow.com/questions/47967549/what-is-the-rationale-for-not-being-able-to-use-the-outer-type-parameter-within
            // can't use Self::CHUNK_SIZE inside this func
            const CHUNK_SIZE: usize = 8;
            let last_dim_elems = tensor.shape.shape[dim];
            let new_shape = tensor.shape.remove_dim(dim);
            let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());

            if last_dim_elems < CHUNK_SIZE {
                tensor.data.chunks_exact(last_dim_elems).for_each(|chunk| {
                    data.push(chunk.iter().sum());
                });
            } else {
                tensor.data.chunks_exact(last_dim_elems).for_each(|chunk| {
                    let mut sum = 0.0;
                    chunk.array_chunks::<CHUNK_SIZE>().for_each(|&oct_chunk| {
                        let aligned = f32x8::from_array(oct_chunk);
                        sum += aligned.reduce_sum();
                    });
                    sum += chunk[(chunk.len() / CHUNK_SIZE) * CHUNK_SIZE..]
                        .iter()
                        .sum::<f32>();
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
                let (_, d_aligned, d_suffix) = data.as_simd_mut::<{ Self::CHUNK_SIZE }>();

                tensor.data.chunks_exact(stride).for_each(|row| {
                    row.array_chunks::<{ Self::CHUNK_SIZE }>()
                        .zip(d_aligned.iter_mut())
                        .for_each(|(&chunk, d_simd)| {
                            let aligned = f32x8::from_array(chunk);
                            *d_simd += aligned;
                        });

                    row[(row.len() / Self::CHUNK_SIZE) * Self::CHUNK_SIZE..]
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
                        let (_, d_aligned, d_suffix) = acc.as_simd_mut::<{ Self::CHUNK_SIZE }>();
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
                                    .array_chunks::<{ Self::CHUNK_SIZE }>()
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
                        let (_, d_aligned, d_suffix) = acc.as_simd_mut::<{ Self::CHUNK_SIZE }>();

                        row.chunks_exact(row_stride).for_each(|chunk| {
                            chunk
                                .array_chunks::<{ Self::CHUNK_SIZE }>()
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
        let (_, a_aligned, a_suffix) = a.data.as_simd::<{ Self::CHUNK_SIZE }>();

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
        let (_, a_aligned, a_suffix) = a.data.as_simd::<{ Self::CHUNK_SIZE }>();

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
        let (_, a_aligned, a_suffix) = a.data.as_simd::<{ Self::CHUNK_SIZE }>();

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
        let (_, a_aligned, a_suffix) = a.data.as_simd::<{ Self::CHUNK_SIZE }>();

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

        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, a.data.len());
        let (_, a_aligned, a_suffix) = a.data.as_simd::<{ Self::CHUNK_SIZE }>();
        let (_, b_aligned, b_suffix) = b.data.as_simd::<{ Self::CHUNK_SIZE }>();

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

        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, a.data.len());
        let (_, a_aligned, a_suffix) = a.data.as_simd::<{ Self::CHUNK_SIZE }>();
        let (_, b_aligned, b_suffix) = b.data.as_simd::<{ Self::CHUNK_SIZE }>();

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

        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, a.data.len());
        let (_, a_aligned, a_suffix) = a.data.as_simd::<{ Self::CHUNK_SIZE }>();
        let (_, b_aligned, b_suffix) = b.data.as_simd::<{ Self::CHUNK_SIZE }>();

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
    fn test_matmul() {
        backend_tests::<AVX2Backend, f32>::test_matmul();
    }

    #[test]
    fn test_exp() {
        backend_tests::<AVX2Backend, f32>::test_exp();
    }

    #[test]
    fn test_log2() {
        backend_tests::<AVX2Backend, f32>::test_log2();
    }

    #[test]
    fn test_relu() {
        backend_tests::<AVX2Backend, f32>::test_relu();
    }

    #[test]
    fn test_sqrt() {
        backend_tests::<AVX2Backend, f32>::test_sqrt();
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
