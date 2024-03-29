use alloc::sync::Arc;
use core::{
    ops::Add,
    simd::{f32x8, num::SimdFloat},
};

use aligned_vec::{avec, AVec};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use super::Backend;
use crate::{Shape, Tensor, CACHELINE_ALIGN};

const CACHE_LINE_F32: usize = 16;

pub struct AVX2Backend;

impl Backend<f32> for AVX2Backend {
    const CHUNK_SIZE: usize = 8;

    fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        // TODO: only supports 2d tensors as of now

        // NOTE:
        // torch is able to matmul two [2048, 2048] in 0.11s / 110ms.
        // our impl takes about 4s.
        // we should at least get to 1s.
        // TODO: find what's causing the bottleneck

        // https://iq.opengenus.org/formula-for-flops-theoretical-max
        // Theoretical Maximum FLOPS = Clock Speed x Number of Cores x SIMD factor x FMA factor x Instuctions per second
        // theoretical flops (multi-thread) on my pc = 3.4 * 6 * 8 * 8 * 1 = 1305.6 GFLOPs
        // theoretical flops (single-thread) on my pc = 3.4 * 1 * 8 * 8 * 1 = 217.6 GFLOPs
        //
        // 2 * N compute for each cell of matrix
        // 2 * N * N * N compute for the whole matrix (N * N cells)
        // for N = 2048, compute = 17179869184
        // TODO: should i also include the compute for copying the values to simd registers to
        // compute?
        //
        // my impl takes 4 secs = 17179869184 / 4 = 4294967296 = 17.179869184 GFLOPs
        // torch impl takes 0.1 secs = 17179869184 / 0.1 = 171798691840 = 171.79869184 GFLOPs
        // numpy imple takes 60 secs = 17179869184 / 60 = 286331153 = 0.286331153 GFLOPs
        // theoretical we can do 2*N^3 compute in 0.079169904s = about 80ms

        assert!(
            a.is_contiguous() && b.is_contiguous(),
            "tensors must be continguous"
        );

        let a_row = a.shape()[0];
        let b_row = b.shape()[0];
        let a_col = a.shape()[1];
        let b_col = b.shape()[1];
        assert_eq!(a_col, b_row);

        let new_shape = Shape::new(&[a_row, b_col]);
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());
        (0..new_shape.numel()).for_each(|_| data.push(0.0));

        let rem = b_col % CACHE_LINE_F32;
        let num_iters = b_col / CACHE_LINE_F32;
        let j_final = b_col - rem;

        const SIZE: usize = 3;
        // const SIZE: usize = 4;

        let get_wide_array = || -> [wide::f32x8; 2] { [wide::f32x8::ZERO; 2] };
        let cast_wide_to_array = |simd_array: [wide::f32x8; 2]| -> [f32; 16] {
            let mut array = [0.0_f32; 16];
            array[..8].copy_from_slice(&simd_array[0].to_array());
            array[8..].copy_from_slice(&simd_array[1].to_array());
            return array;
        };

        for i in (0..((a_row / SIZE) * SIZE)).step_by(SIZE) {
            let row_0 = &a.data[i * a_col..i * a_col + a_col];
            let row_1 = &a.data[(i + 1) * a_col..(i + 1) * a_col + a_col];
            let row_2 = &a.data[(i + 2) * a_col..(i + 2) * a_col + a_col];
            // let row_3 = &a.data[(i + 3) * a_col..(i + 3) * a_col + a_col];

            for j in (0..num_iters).map(|x| x * CACHE_LINE_F32) {
                let mut aligned_0 = get_wide_array();
                let mut aligned_1 = get_wide_array();
                let mut aligned_2 = get_wide_array();
                // let mut aligned_3 = get_wide_array();

                row_0
                    .iter()
                    .zip(row_1)
                    .zip(row_2)
                    // .zip(row_3)
                    .enumerate()
                    // .for_each(|(k, (((&elem_0, &elem_1), &elem_2), &elem_3))| {
                    .for_each(|(k, ((&elem_0, &elem_1), &elem_2))| {
                        let elem_0_simd = wide::f32x8::splat(elem_0);
                        let elem_1_simd = wide::f32x8::splat(elem_1);
                        let elem_2_simd = wide::f32x8::splat(elem_2);
                        // let elem_3_simd = wide::f32x8::splat(elem_3);
                        let col = &b.data[(k * b_col) + j..(k * b_col) + j + CACHE_LINE_F32];

                        aligned_0
                            .iter_mut()
                            .zip(aligned_1.iter_mut())
                            .zip(aligned_2.iter_mut())
                            // .zip(aligned_3.iter_mut())
                            .zip(col.array_chunks::<8>())
                            .for_each(
                                // |((((buf_0_mut, buf_1_mut), buf_2_mut), buf_3_mut), &col_chunk)| {
                                |(((buf_0_mut, buf_1_mut), buf_2_mut), &col_chunk)| {
                                    let col_chunk_simd = wide::f32x8::from(col_chunk);
                                    *buf_0_mut = elem_0_simd.mul_add(col_chunk_simd, *buf_0_mut);
                                    *buf_1_mut = elem_1_simd.mul_add(col_chunk_simd, *buf_1_mut);
                                    *buf_2_mut = elem_2_simd.mul_add(col_chunk_simd, *buf_2_mut);
                                    // *buf_3_mut = elem_3_simd.mul_add(col_chunk_simd, *buf_3_mut);
                                },
                            );
                    });

                data[(i * b_col) + j..(i * b_col) + j + CACHE_LINE_F32]
                    .copy_from_slice(cast_wide_to_array(aligned_0).as_slice());

                data[((i + 1) * b_col) + j..((i + 1) * b_col) + j + CACHE_LINE_F32]
                    .copy_from_slice(cast_wide_to_array(aligned_1).as_slice());

                data[((i + 2) * b_col) + j..((i + 2) * b_col) + j + CACHE_LINE_F32]
                    .copy_from_slice(cast_wide_to_array(aligned_2).as_slice());

                // data[((i + 3) * b_col) + j..((i + 3) * b_col) + j + CACHE_LINE_F32]
                //     .copy_from_slice(cast_wide_to_array(aligned_3).as_slice());
            }

            // let rows = [row_0, row_1, row_2, row_3];
            if rem > 0 {
                let rows = [row_0, row_1, row_2];
                for (k, &row) in rows.iter().enumerate() {
                    let mut buffer = [0.0_f32; CACHE_LINE_F32];
                    row.iter().enumerate().for_each(|(k, &elem)| {
                        let col = &b.data[(k * b_col) + j_final..(k * b_col) + j_final + rem];
                        buffer
                            .iter_mut()
                            .zip(col.iter())
                            .for_each(|(dst, col_elem)| *dst += col_elem * elem);
                    });

                    data[((i + k) * b_col) + j_final..((i + k) * b_col) + j_final + rem]
                        .iter_mut()
                        .zip(&buffer[..rem])
                        .for_each(|(dst, &src)| *dst += src);
                }
            }
        }

        let row_rem = a_row % SIZE;
        if row_rem != 0 {
            (a_row - row_rem..a_row).for_each(|i| {
                let row = &a.data[i * a_col..i * a_col + a_col];
                for j in (0..num_iters).map(|x| x * CACHE_LINE_F32) {
                    let mut aligned = get_wide_array();

                    row.iter().enumerate().for_each(|(k, &elem)| {
                        let elem_simd = wide::f32x8::splat(elem);
                        let col = &b.data[(k * b_col) + j..(k * b_col) + j + CACHE_LINE_F32];

                        aligned.iter_mut().zip(col.array_chunks::<8>()).for_each(
                            |(buf_mut, &col_chunk)| {
                                let col_chunk_simd = wide::f32x8::from(col_chunk);
                                *buf_mut = elem_simd.mul_add(col_chunk_simd, *buf_mut);
                            },
                        );
                    });

                    data[(i * b_col) + j..(i * b_col) + j + CACHE_LINE_F32]
                        .copy_from_slice(cast_wide_to_array(aligned).as_slice());
                }

                let mut buffer = [0.0_f32; CACHE_LINE_F32];
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
            });
        }

        Tensor {
            data: Arc::new(data),
            shape: new_shape,
            name: "matmul".into(),
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
            name: "matmul_naive".into(),
        }
    }

    fn abs(tensor: &Tensor<f32>) -> Tensor<f32> {
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());
        let (_, a_aligned, a_suffix) = tensor.data.as_simd::<{ Self::CHUNK_SIZE }>();

        a_aligned.iter().for_each(|a_vec| {
            a_vec
                .abs()
                .as_array()
                .iter()
                .for_each(|&elem| data.push(elem));
        });
        a_suffix.iter().for_each(|a_elem| {
            data.push(a_elem.abs());
        });

        Tensor {
            data: Arc::new(data),
            shape: tensor.shape,
            name: "abs".into(),
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
            shape: tensor.shape,
            name: "exp".into(),
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
            shape: tensor.shape,
            name: "log2".into(),
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
            shape: tensor.shape,
            name: "relu".into(),
        }
    }

    fn sqrt(tensor: &Tensor<f32>) -> Tensor<f32> {
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());
        tensor
            .data
            .array_chunks::<{ Self::CHUNK_SIZE }>()
            .for_each(|&chunk| {
                let chunk_simd = wide::f32x8::from(chunk);
                let sqrt = chunk_simd.sqrt();
                sqrt.to_array().into_iter().for_each(|val| data.push(val));
            });
        tensor.data[(data.len() / Self::CHUNK_SIZE) * Self::CHUNK_SIZE..]
            .iter()
            .for_each(|val| {
                data.push(val.sqrt());
            });

        Tensor {
            data: Arc::new(data),
            shape: tensor.shape,
            name: "sqrt".into(),
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
            shape: tensor.shape,
            name: "silu".into(),
        }
    }

    fn square(tensor: &Tensor<f32>) -> Tensor<f32> {
        let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, tensor.data.len());
        tensor
            .data
            .array_chunks::<{ Self::CHUNK_SIZE }>()
            .for_each(|&chunk| {
                // TODO: benchmark
                // let square = chunk * chunk;
                let square = wide::f32x8::from(chunk).powf(2.0);
                square.to_array().into_iter().for_each(|val| data.push(val));
            });
        tensor.data[(data.len() / Self::CHUNK_SIZE) * Self::CHUNK_SIZE..]
            .iter()
            .for_each(|val| data.push(val.powi(2)));

        Tensor {
            data: Arc::new(data),
            shape: tensor.shape,
            name: "square".into(),
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
            shape: tensor.shape,
            name: "sigmoid".into(),
        }
    }

    fn sum(tensor: &Tensor<f32>) -> f32 {
        assert!(
            tensor.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        let (_, aligned, suffix) = tensor.data.as_simd::<{ Self::CHUNK_SIZE }>();
        let acc = f32x8::splat(0.0);
        let acc = aligned.iter().fold(acc, f32x8::add);
        acc.reduce_sum() + suffix.iter().sum::<f32>()
    }

    fn sum_rayon(tensor: &Tensor<f32>) -> f32 {
        assert!(
            tensor.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        tensor.data.par_iter().sum::<f32>()
    }

    fn sum_axis(tensor: &Tensor<f32>, dim: usize) -> Tensor<f32> {
        // i'm not really satisfied w this code, kinda feels messy and easy to break also it
        // involves a lot of copying data to simd registers, so i'm kinda skeptical about this one
        // and we'll have to see.
        assert!(
            tensor.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        assert!(
            dim < tensor.shape.ndim,
            "dim must be less than {}",
            tensor.shape.ndim
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
                name: "sum".into(),
            }
        }

        match dim {
            0 => {
                let mut data = avec![0.0_f32; new_shape.numel()];
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
                    name: "sum".into(),
                }
            }
            1 => {
                let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());

                // TODO: using SIMD for 2-D tensors will only add to overhead, cause row_stride would be 1, so will be `stride` chunks
                tensor.data.chunks_exact(stride).for_each(|row| {
                    let mut acc = avec![0.0_f32; row_stride];
                    {
                        let (_, d_aligned, d_suffix) = acc.as_simd_mut::<{ Self::CHUNK_SIZE }>();
                        if d_aligned.is_empty() {
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
                    name: "sum".into(),
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
                        let mut acc = avec![0.0_f32; row_stride];
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
                        name: "sum".into(),
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
            shape: a.shape,
            name: "add_scalar".into(),
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
            shape: a.shape,
            name: "sub_scalar".into(),
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
            shape: a.shape,
            name: "mul_scalar".into(),
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
            shape: a.shape,
            name: "div_scalar".into(),
        }
    }

    fn add_elementwise(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        assert!(
            a.is_contiguous() && b.is_contiguous(),
            "tensors must be continguous"
        );
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
            shape: a.shape,
            name: "add_elementwise".into(),
        }
    }

    fn sub_elementwise(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        assert!(
            a.is_contiguous() && b.is_contiguous(),
            "tensors must be continguous"
        );
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
            shape: a.shape,
            name: "sub_elementwise".into(),
        }
    }

    fn mul_elementwise(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        assert!(
            a.is_contiguous() && b.is_contiguous(),
            "tensors must be continguous"
        );
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
            shape: a.shape,
            name: "mul_elementwise".into(),
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
