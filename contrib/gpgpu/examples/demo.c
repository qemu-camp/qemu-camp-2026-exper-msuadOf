/* SPDX-License-Identifier: MIT */
#include "gpgpu.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

struct args { uint32_t a, b, c, n; };

static void check(int result, const char *operation)
{
    if (result) {
        fprintf(stderr, "%s: %s\n", operation, strerror(-result));
        exit(1);
    }
}

static void verify(const float *actual, const float *expected, unsigned count,
                    const char *name)
{
    for (unsigned i = 0; i < count; i++) {
        if (actual[i] != expected[i]) {
            fprintf(stderr, "%s[%u]: got %g, expected %g\n", name, i,
                    actual[i], expected[i]);
            exit(1);
        }
    }
    printf("%s: PASS (%u results)\n", name, count);
}

int main(int argc, char **argv)
{
    gpgpuDevice *gpu;
    struct args args;
    float a[64], b[64], c[64], expected[64];
    char path[1024];
    const char *dir = argc > 1 ? argv[1] : "/kernels";
    const gpgpuDim3 vector_grid = {2, 1, 1}, vector_block = {32, 1, 1};
    const gpgpuDim3 matrix_grid = {2, 2, 1}, matrix_block = {4, 4, 1};

    check(gpgpuOpen(&gpu, NULL), "open");
    check(gpgpuMalloc(gpu, &args.a, sizeof(a)), "allocate a");
    check(gpgpuMalloc(gpu, &args.b, sizeof(b)), "allocate b");
    check(gpgpuMalloc(gpu, &args.c, sizeof(c)), "allocate c");

    for (unsigned i = 0; i < 64; i++) {
        a[i] = (int)i - 32;
        b[i] = 2 * i;
        expected[i] = a[i] + b[i];
    }
    args.n = 64;
    check(gpgpuMemcpy(gpu, args.a, a, sizeof(a), gpgpuHostToDevice), "upload a");
    check(gpgpuMemcpy(gpu, args.b, b, sizeof(b), gpgpuHostToDevice), "upload b");
    snprintf(path, sizeof(path), "%s/vecadd.bin", dir);
    check(gpgpuLoadKernel(gpu, path), "load vector add");
    check(gpgpuLaunchKernel(gpu, vector_grid, vector_block, &args, sizeof(args)),
          "launch vector add");
    check(gpgpuMemcpy(gpu, args.c, c, sizeof(c), gpgpuDeviceToHost), "download");
    verify(c, expected, 64, "vector add");

    for (unsigned i = 0; i < 64; i++) {
        expected[i] = a[i] > 0 ? a[i] : 0;
    }
    snprintf(path, sizeof(path), "%s/relu.bin", dir);
    check(gpgpuLoadKernel(gpu, path), "load ReLU");
    check(gpgpuLaunchKernel(gpu, vector_grid, vector_block, &args, sizeof(args)),
          "launch ReLU");
    check(gpgpuMemcpy(gpu, args.c, c, sizeof(c), gpgpuDeviceToHost), "download");
    verify(c, expected, 64, "ReLU");

    args.n = 8;
    for (unsigned i = 0; i < 64; i++) {
        a[i] = (int)(i % 11) - 5;
        b[i] = (int)(i % 7) - 3;
    }
    for (unsigned row = 0; row < 8; row++) {
        for (unsigned col = 0; col < 8; col++) {
            float sum = 0;
            for (unsigned k = 0; k < 8; k++) {
                sum += a[row * 8 + k] * b[k * 8 + col];
            }
            expected[row * 8 + col] = sum;
        }
    }
    check(gpgpuMemcpy(gpu, args.a, a, sizeof(a), gpgpuHostToDevice), "upload a");
    check(gpgpuMemcpy(gpu, args.b, b, sizeof(b), gpgpuHostToDevice), "upload b");
    snprintf(path, sizeof(path), "%s/matmul.bin", dir);
    check(gpgpuLoadKernel(gpu, path), "load matrix multiplication");
    check(gpgpuLaunchKernel(gpu, matrix_grid, matrix_block, &args, sizeof(args)),
          "launch matrix multiplication");
    check(gpgpuMemcpy(gpu, args.c, c, sizeof(c), gpgpuDeviceToHost), "download");
    verify(c, expected, 64, "matrix multiplication");

    void *mapped = gpgpuMap(gpu, args.c, 4096);
    if (mapped == MAP_FAILED) {
        perror("map output");
        return 1;
    }
    verify(mapped, expected, 64, "BAR2 mmap");
    munmap(mapped, 4096);
    check(gpgpuDeviceSynchronize(gpu), "synchronize");
    check(gpgpuFree(gpu, args.a), "free a");
    check(gpgpuFree(gpu, args.b), "free b");
    check(gpgpuFree(gpu, args.c), "free c");
    gpgpuClose(gpu);
    puts("GPGPU SDK: ALL DEMOS PASSED");
    return 0;
}
