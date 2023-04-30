# ruml

The goal of this project is to implement a tiny inference only library for
running ML models. I want this to be something like `ggml` and `tinygrad`.

The idea is to support different optimization backends like:

- Accelerate
- AVX
- openblas
- cuBLAS (not sure about cuBLAS)
- naive CPU only (fallback)
- etc

The roadmap right now is more or less like this:

- [ ] implement a minimal tensor class with support for broadcasting and dynamic shapes
- [ ] implement a CPU only backend and write tests for different ops
- [ ] write other backends
- [ ] support fp16, int8 and quantization
- [ ] a demo of the lib using llama or something similar
- [ ] would also like this to work on vision models like segment anything, resnet, etc
