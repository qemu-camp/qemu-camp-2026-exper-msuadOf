/* SPDX-License-Identifier: MIT */
#include "kernel.h"

struct Args { uint32_t a, b, c, n; };

extern "C" __global__ void gpgpu_kernel(const Args *args)
{
    uint32_t i = gpgpu_global_x();
    if (i < args->n) {
        const float *a = (const float *)args->a;
        const float *b = (const float *)args->b;
        float *c = (float *)args->c;
        c[i] = a[i] + b[i];
    }
}
