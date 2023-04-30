use super::Backend;

pub struct CpuBackend {}

impl Backend for CpuBackend {
    fn new() -> Self {
        todo!()
    }

    fn forward(&self) {
        todo!()
    }

    fn matmul(&self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::CpuBackend;

    #[test]
    #[ignore = "unimplemented"]
    fn test_matmul() {
        crate::backend::tests::test_matmul::<CpuBackend>();
    }
}
