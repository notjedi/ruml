use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruml::{AVX2Backend, Backend, Shape, Tensor};

fn bench(c: &mut Criterion) {
    let x_shape = Shape::new(&[2048, 2048]);
    let x = Tensor::<f32>::arange(x_shape.numel()).reshape(x_shape.shape());

    c.bench_function("sum", |b| b.iter(|| AVX2Backend::sum(black_box(&x))));
    c.bench_function("sum_rayon", |b| {
        b.iter(|| AVX2Backend::sum_rayon(black_box(&x)))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
