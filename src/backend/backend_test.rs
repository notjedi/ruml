use core::marker::PhantomData;

use crate::CACHELINE_ALIGN;
use aligned_vec::AVec;

use super::*;

#[allow(non_camel_case_types)]
pub struct tests<T, U>
where
    T: Backend<U>,
    U: NumFloat,
{
    backend: PhantomData<T>,
    dtype: PhantomData<U>,
}

impl<T, U> tests<T, U>
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

    pub fn test_log2() {
        // BUG: doesn't work on the below input, cause the wide crate reports +inf instead of -inf for log2(0)
        // let vals_iter = (1..10).map(|x| U::from(x).unwrap());
        let vals_iter = (1..10).map(|x| U::from(x).unwrap());
        let a = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
        let out = T::log2(&a);
        assert_eq!(
            out.ravel(),
            AVec::from_iter(
                CACHELINE_ALIGN,
                [0.0, 1.0, 1.5849625, 2.0, 2.32192809, 2.5849625, 2.80735492, 3.0, 3.169925]
                    .into_iter()
                    .map(|x| U::from(x).unwrap())
            )
        );
    }

    pub fn test_exp() {
        let vals_iter = (0..10).map(|x| U::from(x).unwrap());
        let a = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
        let out = T::exp(&a);
        assert_eq!(
            out.ravel(),
            AVec::from_iter(
                CACHELINE_ALIGN,
                [
                    1.0, 2.7182817, 7.389056, 20.085537, 54.598152, 148.41316, 403.4288, 1096.6332,
                    2980.958, 8103.084
                ]
                .into_iter()
                .map(|x| U::from(x).unwrap())
            )
        );
    }

    pub fn test_silu() {
        let vals_iter = (0..10).map(|x| U::from(x).unwrap());
        let a = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
        let out = T::silu(&a);
        assert_eq!(
            out.ravel(),
            AVec::from_iter(
                CACHELINE_ALIGN,
                [
                    0.0, 0.7312012, 1.7617188, 2.8582764, 3.9282227, 4.967041, 5.9853516,
                    6.9940186, 7.9973173, 8.99889
                ]
                .into_iter()
                .map(|x| U::from(x).unwrap())
            )
        );
    }

    pub fn test_sigmoid() {
        let vals_iter = (-5..5).map(|x| U::from(x).unwrap());
        let a = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
        let out = T::sigmoid(&a);
        assert_eq!(
            out.ravel(),
            AVec::from_iter(
                CACHELINE_ALIGN,
                [
                    0.0066928864,
                    0.017986298,
                    0.047431946,
                    0.11920166,
                    0.2689209,
                    0.49987793,
                    0.7312012,
                    0.8808594,
                    0.95257413,
                    0.98201376
                ]
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

    pub fn test_sum_axis() {
        // TODO: test on more shapes and different dims
        let shape = [2, 3, 4, 5];
        let len = shape.iter().product();
        let tensor = Tensor::<U>::arange(len).reshape(&shape);

        // dim 0
        let out = T::sum_axis(&tensor, 0);
        [
            60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100,
            102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134,
            136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168,
            170, 172, 174, 176, 178,
        ]
        .iter()
        .zip(out.data.iter())
        .for_each(|(&g_truth, &res)| {
            assert_eq!(U::from(g_truth).unwrap(), res);
        });

        // dim 1
        let out = T::sum_axis(&tensor, 1);
        [
            60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117,
            240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270, 273, 276, 279, 282, 285, 288,
            291, 294, 297,
        ]
        .iter()
        .zip(out.data.iter())
        .for_each(|(&g_truth, &res)| {
            assert_eq!(U::from(g_truth).unwrap(), res);
        });

        // dim 2
        let out = T::sum_axis(&tensor, 2);
        [
            30, 34, 38, 42, 46, 110, 114, 118, 122, 126, 190, 194, 198, 202, 206, 270, 274, 278,
            282, 286, 350, 354, 358, 362, 366, 430, 434, 438, 442, 446,
        ]
        .iter()
        .zip(out.data.iter())
        .for_each(|(&g_truth, &res)| {
            assert_eq!(U::from(g_truth).unwrap(), res);
        });

        // dim 3
        let out = T::sum_axis(&tensor, 3);
        [
            10, 35, 60, 85, 110, 135, 160, 185, 210, 235, 260, 285, 310, 335, 360, 385, 410, 435,
            460, 485, 510, 535, 560, 585,
        ]
        .iter()
        .zip(out.data.iter())
        .for_each(|(&g_truth, &res)| {
            assert_eq!(U::from(g_truth).unwrap(), res);
        });
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
