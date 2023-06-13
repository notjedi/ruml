use aligned_vec::{avec, AVec, CACHELINE_ALIGN};

use super::Backend;
use crate::Tensor;
use core::simd::f32x8;

pub struct CpuBackend {}

// will this work for neon also? cause f32x8::from_slice_unaligned is using llvm codegen, i think
// it does, so i'll comment out the avx target_feature part for now
// #[cfg(all(
//     any(target_arch = "x86", target_arch = "x86_64"),
//     target_feature = "avx2"
// ))]
impl Backend<f32> for CpuBackend {
    fn matmul() {
        todo!()
    }

    fn sum(tensor: &Tensor<f32>) -> f32 {
        use core::simd::SimdFloat;
        use std::ops::Add;

        debug_assert!(
            tensor.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        // TODO: as_simd does not guarantee prefix.len() and suffix.len() to be < LANES, will this affect performance?
        dbg!(tensor.data.len());
        let (prefix, aligned, suffix) = tensor.data.as_simd::<8>();
        let acc = f32x8::from_array([
            prefix.iter().sum::<f32>(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            suffix.iter().sum::<f32>(),
        ]);
        let acc = aligned.iter().fold(acc, f32x8::add);
        acc.reduce_sum()
    }

    fn sum_axis(tensor: &Tensor<f32>, dim: usize) -> Tensor<f32> {
        debug_assert!(
            tensor.is_contiguous(),
            "vector instructions are only supported for contiguous tensors"
        );
        let stride = tensor.shape.strides[0];
        let total_rows = tensor.shape.shape[0];
        let row_stride = tensor.shape.strides[dim];
        let new_shape = tensor.shape.remove_dim(dim);

        match dim {
            0 => {
                let mut data = avec![0.0 as f32; new_shape.numel()];
                let (d_prefix, d_aligned, d_suffix) = data.as_simd_mut::<8>();
                assert!(
                    d_prefix.len() == 0,
                    "bro, something is wrong w your code, check alignment of data"
                );

                (0..total_rows).for_each(|idx| {
                    // let row = &tensor.data[idx * row_stride..(idx+1) * row_stride];
                    let row = &tensor.data[idx * stride..idx * stride + stride];
                    let (prefix, aligned, suffix) = row.as_simd::<8>();
                    assert!(
                        prefix.len() == 0,
                        "bro, something is wrong w your code, check alignment of data"
                    );

                    aligned
                        .iter()
                        .zip(d_aligned.iter_mut())
                        .for_each(|(&row_data, orig_data)| *orig_data += row_data);

                    suffix
                        .iter()
                        .zip(d_suffix.iter_mut())
                        .for_each(|(&row_data, orig_data)| *orig_data += row_data);
                });
                todo!()
            }
            1 => {
                let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());

                // TODO: using SIMD for 2-D tensors will only add to overhead, cause row_stride would be 1, so will be `stride` chunks
                (0..total_rows).for_each(|idx| {
                    let row = &tensor.data[idx * stride..idx * stride + stride];
                    let mut acc = avec![0.0 as f32; row_stride];

                    {
                        let (d_prefix, d_aligned, d_suffix) = acc.as_simd_mut::<8>();
                        assert!(
                            d_prefix.len() == 0,
                            "bro, something is wrong w your code, check alignment of data"
                        );

                        row.chunks(row_stride).for_each(|chunk| {
                            let (prefix, aligned, suffix) = chunk.as_simd::<8>();
                            assert!(
                                prefix.len() == 0,
                                "bro, something is wrong w your code, check alignment of data"
                            );
                            aligned
                                .iter()
                                .zip(d_aligned.iter_mut())
                                .for_each(|(&row_data, orig_data)| *orig_data += row_data);
                            suffix
                                .iter()
                                .zip(d_suffix.iter_mut())
                                .for_each(|(&row_data, orig_data)| *orig_data += row_data);
                        });
                    }
                    acc.iter().for_each(|&val| data.push(val));
                });
                todo!()
            }
            2 => {
                use core::simd::SimdFloat;
                use std::ops::Add;

                let mut data = AVec::<f32>::with_capacity(CACHELINE_ALIGN, new_shape.numel());

                // TODO: using SIMD for 3-D tensors will only add to overhead, cause row_stride would be 1, so will be `stride` chunks
                (0..total_rows).for_each(|idx| {
                    let row = &tensor.data[idx * stride..idx * stride + stride];
                    row.chunks(row_stride).for_each(|chunk| {
                        let (prefix, aligned, suffix) = chunk.as_simd::<8>();
                        assert!(
                            prefix.len() == 0,
                            "bro, something is wrong w your code, check alignment of data"
                        );
                        let acc = f32x8::splat(0.0 as f32);
                        let sum = aligned.iter().fold(acc, f32x8::add);
                        data.push(sum.reduce_sum() + suffix.iter().sum::<f32>());
                    });
                });
                todo!()
            }
            _ => {
                todo!()
            }
        }
    }

    fn add() {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::CpuBackend;
    use crate::backend::tests as backend_tests;

    #[test]
    #[ignore = "unimplemented"]
    fn test_matmul() {
        backend_tests::test_matmul::<CpuBackend>();
    }

    #[test]
    fn test_sum() {
        backend_tests::test_sum::<CpuBackend>();
    }
}
