use core::marker::PhantomData;

use crate::{Shape, CACHELINE_ALIGN};
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
        let a_shape = Shape::new(&[18, 18]);
        let b_shape = Shape::new(&[18, 18]);
        let a = Tensor::<U>::arange(a_shape.numel()).reshape(a_shape.shape());
        let b = Tensor::<U>::arange(b_shape.numel()).reshape(b_shape.shape());

        let out = T::matmul(&a, &b);
        // let out_naive = T::matmul_naive(&a, &b);

        assert_eq!(
            out.ravel().as_slice(),
            [
                32130, 32283, 32436, 32589, 32742, 32895, 33048, 33201, 33354, 33507, 33660, 33813,
                33966, 34119, 34272, 34425, 34578, 34731, 81702, 82179, 82656, 83133, 83610, 84087,
                84564, 85041, 85518, 85995, 86472, 86949, 87426, 87903, 88380, 88857, 89334, 89811,
                131274, 132075, 132876, 133677, 134478, 135279, 136080, 136881, 137682, 138483,
                139284, 140085, 140886, 141687, 142488, 143289, 144090, 144891, 180846, 181971,
                183096, 184221, 185346, 186471, 187596, 188721, 189846, 190971, 192096, 193221,
                194346, 195471, 196596, 197721, 198846, 199971, 230418, 231867, 233316, 234765,
                236214, 237663, 239112, 240561, 242010, 243459, 244908, 246357, 247806, 249255,
                250704, 252153, 253602, 255051, 279990, 281763, 283536, 285309, 287082, 288855,
                290628, 292401, 294174, 295947, 297720, 299493, 301266, 303039, 304812, 306585,
                308358, 310131, 329562, 331659, 333756, 335853, 337950, 340047, 342144, 344241,
                346338, 348435, 350532, 352629, 354726, 356823, 358920, 361017, 363114, 365211,
                379134, 381555, 383976, 386397, 388818, 391239, 393660, 396081, 398502, 400923,
                403344, 405765, 408186, 410607, 413028, 415449, 417870, 420291, 428706, 431451,
                434196, 436941, 439686, 442431, 445176, 447921, 450666, 453411, 456156, 458901,
                461646, 464391, 467136, 469881, 472626, 475371, 478278, 481347, 484416, 487485,
                490554, 493623, 496692, 499761, 502830, 505899, 508968, 512037, 515106, 518175,
                521244, 524313, 527382, 530451, 527850, 531243, 534636, 538029, 541422, 544815,
                548208, 551601, 554994, 558387, 561780, 565173, 568566, 571959, 575352, 578745,
                582138, 585531, 577422, 581139, 584856, 588573, 592290, 596007, 599724, 603441,
                607158, 610875, 614592, 618309, 622026, 625743, 629460, 633177, 636894, 640611,
                626994, 631035, 635076, 639117, 643158, 647199, 651240, 655281, 659322, 663363,
                667404, 671445, 675486, 679527, 683568, 687609, 691650, 695691, 676566, 680931,
                685296, 689661, 694026, 698391, 702756, 707121, 711486, 715851, 720216, 724581,
                728946, 733311, 737676, 742041, 746406, 750771, 726138, 730827, 735516, 740205,
                744894, 749583, 754272, 758961, 763650, 768339, 773028, 777717, 782406, 787095,
                791784, 796473, 801162, 805851, 775710, 780723, 785736, 790749, 795762, 800775,
                805788, 810801, 815814, 820827, 825840, 830853, 835866, 840879, 845892, 850905,
                855918, 860931, 825282, 830619, 835956, 841293, 846630, 851967, 857304, 862641,
                867978, 873315, 878652, 883989, 889326, 894663, 900000, 905337, 910674, 916011,
                874854, 880515, 886176, 891837, 897498, 903159, 908820, 914481, 920142, 925803,
                931464, 937125, 942786, 948447, 954108, 959769, 965430, 971091
            ]
            .into_iter()
            .map(|x| U::from(x).unwrap())
            .collect::<Vec<U>>()
        );
    }

    pub fn test_exp() {
        let vals_iter = (0..10).map(|x| U::from(x).unwrap());
        let a = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
        let out = T::exp(&a);
        assert_eq!(
            out.ravel().as_slice(),
            [
                1.0, 2.7182817, 7.389056, 20.085537, 54.598152, 148.41316, 403.4288, 1096.6332,
                2980.958, 8103.084
            ]
            .into_iter()
            .map(|x| U::from(x).unwrap())
            .collect::<Vec<U>>()
        );
    }

    pub fn test_log2() {
        // BUG: doesn't work on the below input, cause `wide` crate reports +inf instead of -inf for log2(0)
        // let vals_iter = (0..10).map(|x| U::from(x).unwrap());
        let vals_iter = (1..10).map(|x| U::from(x).unwrap());
        let a = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
        let out = T::log2(&a);
        assert_eq!(
            out.ravel().as_slice(),
            [0.0, 1.0, 1.5849625, 2.0, 2.32192809, 2.5849625, 2.80735492, 3.0, 3.169925]
                .into_iter()
                .map(|x| U::from(x).unwrap())
                .collect::<Vec<U>>()
        );
    }

    pub fn test_relu() {
        let vals_iter = (-5..5).map(|x| U::from(x).unwrap());
        let a = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
        let out = T::relu(&a);
        assert_eq!(
            out.ravel().as_slice(),
            [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
                .into_iter()
                .map(|x| U::from(x).unwrap())
                .collect::<Vec<U>>()
        );
    }

    pub fn test_sqrt() {
        let vals_iter = (0..5).map(|x| U::from(x).unwrap());
        let a = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
        let out = T::sqrt(&a);
        assert_eq!(
            out.ravel().as_slice(),
            [0.0, 1.0, 1.4142135, 1.7320508, 2.0]
                .into_iter()
                .map(|x| U::from(x).unwrap())
                .collect::<Vec<U>>()
        );
    }

    pub fn test_silu() {
        let vals_iter = (0..10).map(|x| U::from(x).unwrap());
        let a = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
        let out = T::silu(&a);
        assert_eq!(
            out.ravel().as_slice(),
            [
                0.0, 0.7312012, 1.7617188, 2.8582764, 3.9282227, 4.967041, 5.9853516, 6.9940186,
                7.9973173, 8.99889
            ]
            .into_iter()
            .map(|x| U::from(x).unwrap())
            .collect::<Vec<U>>()
        );
    }

    pub fn test_sigmoid() {
        let vals_iter = (-5..5).map(|x| U::from(x).unwrap());
        let a = Tensor::new(AVec::from_iter(CACHELINE_ALIGN, vals_iter));
        let out = T::sigmoid(&a);
        assert_eq!(
            out.ravel().as_slice(),
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
            .collect::<Vec<U>>()
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
