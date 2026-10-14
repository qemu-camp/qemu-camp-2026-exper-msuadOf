/* SPDX-License-Identifier: MIT */
#ifndef LIBGPGPU_H
#define LIBGPGPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpgpu_device gpgpuDevice;
typedef uint32_t gpgpuDevicePtr;
typedef struct gpgpu_dim3 { uint32_t x, y, z; } gpgpuDim3;
typedef enum { gpgpuHostToDevice, gpgpuDeviceToHost } gpgpuMemcpyKind;

int gpgpuOpen(gpgpuDevice **device, const char *path);
void gpgpuClose(gpgpuDevice *device);
int gpgpuMalloc(gpgpuDevice *device, gpgpuDevicePtr *pointer, size_t size);
int gpgpuFree(gpgpuDevice *device, gpgpuDevicePtr pointer);
int gpgpuMemcpy(gpgpuDevice *device, gpgpuDevicePtr pointer,
                void *host, size_t size, gpgpuMemcpyKind direction);
int gpgpuLoadKernel(gpgpuDevice *device, const char *image);
int gpgpuLaunchKernel(gpgpuDevice *device, gpgpuDim3 grid, gpgpuDim3 block,
                      const void *args, size_t args_size);
int gpgpuDeviceSynchronize(gpgpuDevice *device);
void *gpgpuMap(gpgpuDevice *device, gpgpuDevicePtr pointer, size_t size);

#ifdef __cplusplus
}
#endif
#endif
