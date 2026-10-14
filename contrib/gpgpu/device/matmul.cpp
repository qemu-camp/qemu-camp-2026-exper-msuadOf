/* SPDX-License-Identifier: MIT */
#include "kernel.h"

struct Args { uint32_t a, b, c, n; };

extern "C" __global__ void gpgpu_kernel(const Args *args)
{
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = gpgpu_global_x();
    if (row < args->n && col < args->n) {
        const float *a = (const float *)args->a;
        const float *b = (const float *)args->b;
        float *c = (float *)args->c;
        float sum = 0;
        for (uint32_t k = 0; k < args->n; k++) {
            sum += a[row * args->n + k] * b[k * args->n + col];
        }
        c[row * args->n + col] = sum;
    }
}
