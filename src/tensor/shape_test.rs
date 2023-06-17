use crate::Shape;

#[test]
fn test_shape() {
    let shape_vec = vec![3, 2, 5, 1];
    let shape: Shape = shape_vec.clone().into();

    assert_eq!(shape.ndim(), 4);
    assert_eq!(shape.shape(), &shape_vec);
    assert_eq!(shape.strides(), &[10, 5, 1, 1]);
    assert_eq!(shape.numel(), shape_vec.iter().product());
    assert_eq!(shape.is_valid_index(&[2, 1, 3, 0]), true);
    assert_eq!(shape.is_valid_index(&[10, 3, 0, 10]), false);
}

#[test]
fn test_shape_ops() {
    // TODO: add tests for other ops like transpose, reduce_dim, etc
    let shape_vec = vec![3, 2, 5, 1];
    let shape: Shape = shape_vec.clone().into();

    let remove_shape = shape.remove_dim(1);
    assert_eq!(remove_shape.shape(), &[3, 5, 1]);
    assert_eq!(remove_shape.strides(), &[10, 1, 1]);

    let squeeze_shape = shape.squeeze();
    assert_eq!(squeeze_shape.shape(), &[3, 2, 5]);
    assert_eq!(squeeze_shape.strides(), &[10, 5, 1]);

    let perm_shape = shape.permute(&[3, 2, 1, 0]);
    assert_eq!(perm_shape.shape(), &[1, 5, 2, 3]);
    assert_eq!(perm_shape.strides(), &[1, 1, 5, 10]);

    let trans_shape = shape.transpose(0, 3);
    assert_eq!(trans_shape.shape(), &[1, 2, 5, 3]);
    assert_eq!(trans_shape.strides(), &[1, 5, 1, 10]);
}

#[test]
fn test_attempt_reshape_without_copying() {
    // normal shape = contiguous
    let shape = Shape {
        shape: vec![4, 3, 2],
        strides: vec![6, 2, 1],
        offset: 0,
    };
    let attempt_reshape = shape.attempt_reshape_without_copying(&[4, 1, 3, 2]);
    assert_eq!(attempt_reshape.unwrap().strides, &[6, 6, 2, 1]);

    // normal shape = contiguous
    let shape = Shape {
        shape: vec![3, 27],
        strides: vec![27, 1],
        offset: 0,
    };
    let attempt_reshape = shape.attempt_reshape_without_copying(&[3, 3, 3, 3]);
    assert_eq!(attempt_reshape.unwrap().strides, &[27, 9, 3, 1]);

    // normal shape = contiguous, with trailing 1's in new_shape
    let shape = Shape {
        shape: vec![3, 27],
        strides: vec![27, 1],
        offset: 0,
    };
    let attempt_reshape = shape.attempt_reshape_without_copying(&[3, 3, 3, 3, 1]);
    assert_eq!(attempt_reshape.unwrap().strides, &[27, 9, 3, 1, 1]);

    // expanded at dim 0
    let shape = Shape {
        shape: vec![8, 5],
        strides: vec![0, 1],
        offset: 0,
    };
    let attempt_reshape = shape.attempt_reshape_without_copying(&[4, 2, 5]);
    assert_eq!(attempt_reshape.unwrap().strides, &[0, 0, 1]);

    // expanded at dim 0
    let shape = Shape {
        shape: vec![3, 6],
        strides: vec![0, 1],
        offset: 0,
    };
    let attempt_reshape = shape.attempt_reshape_without_copying(&[2, 3, 3]);
    assert!(attempt_reshape.is_err());

    // expanded at dim 0
    let shape = Shape {
        shape: vec![3, 6],
        strides: vec![0, 1],
        offset: 0,
    };
    let attempt_reshape = shape.attempt_reshape_without_copying(&[3, 2, 3]);
    assert_eq!(attempt_reshape.unwrap().strides, &[0, 3, 1]);

    // expanded at dim 0
    let shape = Shape {
        shape: vec![6],
        strides: vec![0],
        offset: 0,
    };
    let attempt_reshape = shape.attempt_reshape_without_copying(&[1, 1, 6]);
    assert_eq!(attempt_reshape.unwrap().strides, &[0, 0, 0]);

    // expanded at dim 1
    let shape = Shape {
        shape: vec![4, 3, 2],
        strides: vec![2, 0, 1],
        offset: 0,
    };
    let attempt_reshape = shape.attempt_reshape_without_copying(&[4, 3, 1, 2]);
    assert_eq!(attempt_reshape.unwrap().strides, &[2, 0, 2, 1]);

    // transpose or permute
    let shape = Shape {
        shape: vec![4, 3, 2],
        strides: vec![6, 2, 1],
        offset: 0,
    }
    .permute(&[2, 1, 0]);
    let attempt_reshape = shape.attempt_reshape_without_copying(&[4, 1, 3, 2]);
    assert!(attempt_reshape.is_err());
}
