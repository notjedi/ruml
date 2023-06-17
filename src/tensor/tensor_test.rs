use rand::{rngs::StdRng, SeedableRng};

use super::*;

#[test]
fn test_tensor() {
    let shape_vec = vec![2, 2, 2];
    let shape: Shape = shape_vec.clone().into();

    let ones_tensor: Tensor<f32> = Tensor::ones(&shape_vec);
    assert_eq!(ones_tensor.shape(), shape.shape());
    assert_eq!(
        ones_tensor.ravel(),
        avec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "elements don't match for the Tensor::ones tensor"
    );

    let zeros_tensor: Tensor<f32> = Tensor::zeros(&shape_vec);
    assert_eq!(zeros_tensor.shape(), shape.shape());
    assert_eq!(
        zeros_tensor.ravel(),
        avec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "elements don't match for the Tensor::zeros tensor"
    );

    let arange_tensor: Tensor<f32> = Tensor::arange(shape.numel()).reshape(&shape_vec);
    assert_eq!(arange_tensor.data.len(), shape.numel());
    assert_eq!(arange_tensor.shape(), shape.shape());
    assert_eq!(
        arange_tensor.ravel(),
        avec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "elements don't match for the Tensor::arange tensor"
    );
}

#[test]
fn test_tensor_iter() {
    // TODO: test tensor iter after shape ops like transpose, expand, etc
    let shape_vec = vec![2, 2, 2];

    let ones_tensor: Tensor<f32> = Tensor::ones(&shape_vec);
    ones_tensor
        .into_iter()
        .zip(ones_tensor.data.iter())
        .enumerate()
        .for_each(|(i, (iter, &vec))| {
            assert_eq!(iter, vec, "values differ at {i}");
        });
}

#[test]
fn test_tensor_dim_iter() {
    let shape_vec = vec![1, 2, 3];
    let shape: Shape = shape_vec.clone().into();
    let arange_tensor: Tensor<f32> = Tensor::arange(shape.numel()).reshape(&shape_vec);

    // should return the following elements [0, 1, 2, 3, 4, 5]
    for dim_tensor in arange_tensor.dim_iter(0) {
        assert_eq!(
            dim_tensor.into_iter().collect::<Vec<_>>().as_slice(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }

    // should return the following elements [[0, 1, 2], [3, 4, 5]]
    let mut num = 0.0;
    for dim_tensor in arange_tensor.dim_iter(1) {
        assert_eq!(
            dim_tensor.into_iter().collect::<Vec<_>>().as_slice(),
            vec![num, num + 1.0, num + 2.0]
        );
        num += 3.0;
    }

    // should return the following elements [[0, 3], [1, 4], [2, 5]]
    for (i, dim_tensor) in arange_tensor.dim_iter(2).enumerate() {
        assert_eq!(
            dim_tensor.into_iter().collect::<Vec<_>>().as_slice(),
            vec![i as f32, i as f32 + 3.0]
        );
    }
}

#[test]
fn test_tensor_ops() {
    let add_shape: Shape = vec![2, 2, 2].into();
    let add_tensor = Tensor::arange(add_shape.numel()).reshape(&add_shape.shape) + 5.0;
    assert_eq!(
        add_tensor.ravel(),
        avec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "elements don't match for the Tensor::arange tensor"
    );
}

// #[test]
// fn test_tensor_shape_ops() {
//     let shape: Shape = vec![2, 1, 3].into();
//     let tensor: Tensor<f32> = Tensor::arange(shape.numel())
//         .reshape(&shape.shape)
//         .expand(1, 4);
//     let tensor: Vec<_> = tensor.flatten();
//     TODO: add tests for flatten after expanding
//     // assert!(false);
// }

#[test]
fn test_tensor_mlops() {
    let shape_vec = vec![3, 4, 5];
    let shape: Shape = shape_vec.clone().into();
    let tensor: Tensor<f32> = Tensor::arange(shape.numel()).reshape(&shape_vec);

    let sum_tensor = tensor.sum(None);
    let sum = (shape.numel() * (shape.numel() - 1) / 2) as f32;
    let sum_tensor_check: Tensor<f32> = Tensor::new(avec![sum]);
    assert_eq!(sum_tensor, sum_tensor_check);

    let sum_tensor = tensor.sum(2);
    // import numpy as np; shape = [3, 4, 5]; np.arange(np.prod(shape), dtype=np.float32).reshape(shape).sum(axis=2).flatten()
    let sum_vec: AVec<f32> =
        avec![10.0, 35.0, 60.0, 85.0, 110.0, 135.0, 160.0, 185.0, 210.0, 235.0, 260.0, 285.0];
    assert_eq!(sum_tensor.shape(), &[3, 4, 1]);
    assert_eq!(Arc::try_unwrap(sum_tensor.data).unwrap(), sum_vec);
}

#[test]
fn test_tensor_broadcast() {
    let x = Tensor::<f32>::zeros(&[1, 3]);
    let y = Tensor::<f32>::ones(&[3, 1]);
    let broadcast_tensor = x + y;
    assert_eq!(broadcast_tensor.shape(), &[3, 3]);
    assert_eq!(broadcast_tensor.ravel(), avec![1.0; 9]);

    let x = Tensor::<f32>::zeros(&[5, 3, 1]);
    let y = Tensor::<f32>::ones(&[3, 1]);
    let broadcast_tensor = &x + &y;
    assert_eq!(broadcast_tensor.shape(), &[5, 3, 1]);
    assert_eq!(broadcast_tensor.ravel(), avec![1.0; 15]);

    let broadcast_tensor = &y + &x;
    assert_eq!(broadcast_tensor.shape(), &[5, 3, 1]);
    assert_eq!(broadcast_tensor.ravel(), avec![1.0; 15]);
}

#[test]
fn test_randn() {
    let mut rng = StdRng::seed_from_u64(0u64);
    let tensor = Tensor::<f32>::randn(&[3, 2], &mut rng);
    assert_eq!(tensor.shape(), &[3, 2]);
    assert_eq!(
        tensor.ravel(),
        avec![0.712813, 0.85833144, -2.4362438, 0.16334426, -1.2750102, 1.287171]
    );
}
