use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruml::{AVX2Backend, Backend, Shape, Tensor};

fn bench(c: &mut Criterion) {
    let x_shape = Shape::new(&[1024, 1024]);
    let y_shape = Shape::new(&[1024, 1024]);
    let x = Tensor::<f32>::arange(x_shape.numel()).reshape(x_shape.shape());
    let y = Tensor::<f32>::arange(y_shape.numel()).reshape(y_shape.shape());

    c.bench_function("matmul", |b| {
        b.iter(|| AVX2Backend::matmul(black_box(&x), black_box(&y)))
    });

    c.bench_function("matmul_block", |b| {
        b.iter(|| AVX2Backend::matmul_block(black_box(&x), black_box(&y)))
    });
}

// https://medium.com/@yamafaktory/rust-benchmarking-with-criterion-on-travis-ci-%EF%B8%8F-8b54d321e05
criterion_group!(benches, bench);
criterion_main!(benches);
