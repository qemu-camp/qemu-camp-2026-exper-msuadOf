/* SPDX-License-Identifier: MIT */
#ifndef GPGPU_KERNEL_H
#define GPGPU_KERNEL_H
#include <stdint.h>

#define __global__ __attribute__((used))
typedef struct { uint32_t x, y, z; } gpgpu_index;

static inline gpgpu_index gpgpu_context(uint32_t offset)
{
    volatile uint32_t *ctrl = (volatile uint32_t *)(0x80000000u + offset);
    return (gpgpu_index){ctrl[0], ctrl[1], ctrl[2]};
}

#define threadIdx (gpgpu_context(0x00))
#define blockIdx  (gpgpu_context(0x10))
#define blockDim  (gpgpu_context(0x20))
#define gridDim   (gpgpu_context(0x30))

static inline uint32_t gpgpu_global_x(void)
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

#endif
