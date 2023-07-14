use ruml::{AVX2Backend, Backend, Shape, Tensor};

fn main() {
    let x_shape = Shape::new(&[2048, 2048]);
    let y_shape = Shape::new(&[2048, 2048]);

    let x = Tensor::<f32>::arange(x_shape.numel()).reshape(x_shape.shape());
    let y = Tensor::<f32>::arange(y_shape.numel()).reshape(y_shape.shape());

    let out = AVX2Backend::matmul(&x, &y);
    println!("{}", &out);
}
