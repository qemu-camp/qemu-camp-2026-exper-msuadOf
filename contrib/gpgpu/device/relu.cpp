/* SPDX-License-Identifier: MIT */
#include "kernel.h"

struct Args { uint32_t a, b, c, n; };

extern "C" __global__ void gpgpu_kernel(const Args *args)
{
    uint32_t i = gpgpu_global_x();
    if (i < args->n) {
        const float *a = (const float *)args->a;
        float *c = (float *)args->c;
        float value = a[i];
        c[i] = value > 0 ? value : 0;
    }
}
