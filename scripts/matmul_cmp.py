import time

import numpy as np
import torch

if __name__ == "__main__":
    shape = (2048, 2048)
    numel = np.prod(shape).item()
    torch.set_printoptions(sci_mode=False)

    # row = np.arange(numel).reshape(shape)[0, :16]
    # col = np.arange(numel).reshape(shape)[:16, :16]
    # print((row @ col))

    # numpy
    x = np.arange(numel).reshape(shape)
    y = np.arange(numel).reshape(shape)

    start = time.perf_counter()
    matmul = x @ y
    end = time.perf_counter()
    print("numpy: ", end - start)

    # torch cpu
    x_f32 = torch.arange(numel, dtype=torch.float32).reshape(shape)
    y_f32 = torch.arange(numel, dtype=torch.float32).reshape(shape)

    start = time.perf_counter()
    matmul_f32 = x_f32 @ y_f32
    end = time.perf_counter()
    print("torch (cpu): ", end - start)

    # torch cuda
    x_f32_cuda = torch.arange(numel, dtype=torch.float32, device="cuda").reshape(shape)
    y_f32_cuda = torch.arange(numel, dtype=torch.float32, device="cuda").reshape(shape)

    start = time.perf_counter()
    matmul_f32 = x_f32_cuda @ y_f32_cuda
    end = time.perf_counter()
    print("torch (cuda): ", end - start)
