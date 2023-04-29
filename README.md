# ruml

The goal of this project is to implement a tiny inference only library for
running ML models. I want this to be something like `ggml` and `tinygrad`.

The idea is to support different optimization backends like:

- Accelerate
- AVX
- openblas
- cuBLAS (not sure about cuBLAS)
- CPU (fallback)
- etc

and also to support different ops like:

- NONE
- DUP
- ADD
- SUB
- MUL
- DIV
- SQR
- SQRT
- SUM
- MEAN
- REPEAT
- ABS
- SGN
- NEG
- STEP
- RELU
- GELU
- SILU
- NORM
- RMS_NORM
- MUL_MAT
- SCALE
- CPY
- CONT
- RESHAPE
- VIEW
- PERMUTE
- TRANSPOSE
- GET_ROWS
- DIAG_MASK_INF
- SOFT_MAX
- ROPE
- CONV_1D_1S
- CONV_1D_2S
- FLASH_ATTN
- FLASH_FF
- MAP_UNARY
- MAP_BINARY

The roadmap right now is more or less like this:

- Implement the first CPU only backend and write tests for different ops
- write tests to test every single op
- continue writing other backends and run tests
- support fp16, int8 and quantization
- a demo of the lib using llama or something similar
- would also like this to work on vision models like segment anything, resnet, etc
