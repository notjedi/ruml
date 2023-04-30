pub mod cpu;

pub use cpu::CpuBackend;

pub trait Backend {
    fn new() -> Self;
    fn forward(&self);
    fn matmul(&self);
}

// https://users.rust-lang.org/t/unit-tests-for-traits/86848
pub mod tests {
    use super::*;

    pub fn test_matmul<T: Backend>() {
        let cpu_backend = T::new();
        let out = cpu_backend.matmul();
        assert_eq!(out, ());
    }
}
