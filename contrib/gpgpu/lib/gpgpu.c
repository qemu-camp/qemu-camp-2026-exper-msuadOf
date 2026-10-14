/* SPDX-License-Identifier: MIT */
#include "gpgpu.h"
#include "gpgpu_ioctl.h"
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct gpgpu_device { int fd; };

static int request(gpgpuDevice *device, unsigned long command, void *arg)
{
    if (!device) {
        return -EINVAL;
    }
    return ioctl(device->fd, command, arg) < 0 ? -errno : 0;
}

int gpgpuOpen(gpgpuDevice **device, const char *path)
{
    struct gpgpu_info info;
    gpgpuDevice *d;
    int result;

    if (!device) {
        return -EINVAL;
    }
    *device = NULL;
    d = calloc(1, sizeof(*d));
    if (!d) {
        return -ENOMEM;
    }
    d->fd = open(path ? path : "/dev/gpgpu0", O_RDWR | O_CLOEXEC);
    if (d->fd < 0) {
        result = -errno;
        free(d);
        return result;
    }
    result = request(d, GPGPU_IOCTL_INFO, &info);
    if (!result && info.abi_version != GPGPU_ABI_VERSION) {
        result = -EPROTO;
    }
    if (result) {
        gpgpuClose(d);
        return result;
    }
    *device = d;
    return 0;
}

void gpgpuClose(gpgpuDevice *device)
{
    if (device) {
        close(device->fd);
        free(device);
    }
}

int gpgpuMalloc(gpgpuDevice *device, gpgpuDevicePtr *pointer, size_t size)
{
    struct gpgpu_buffer buffer = { .size = size };
    int result;

    if (!pointer || !size || size > UINT32_MAX) {
        return -EINVAL;
    }
    result = request(device, GPGPU_IOCTL_ALLOC, &buffer);
    if (!result) {
        *pointer = buffer.device_addr;
    }
    return result;
}

int gpgpuFree(gpgpuDevice *device, gpgpuDevicePtr pointer)
{
    struct gpgpu_buffer buffer = { .device_addr = pointer };

    return request(device, GPGPU_IOCTL_FREE, &buffer);
}

int gpgpuMemcpy(gpgpuDevice *device, gpgpuDevicePtr pointer,
                void *host, size_t size, gpgpuMemcpyKind direction)
{
    struct gpgpu_copy copy = {
        .host_addr = (uintptr_t)host,
        .device_addr = pointer,
        .size = size,
        .from_device = direction == gpgpuDeviceToHost,
    };

    if (!host || size > UINT32_MAX ||
        (direction != gpgpuHostToDevice && direction != gpgpuDeviceToHost)) {
        return -EINVAL;
    }
    return request(device, GPGPU_IOCTL_COPY, &copy);
}

int gpgpuLoadKernel(gpgpuDevice *device, const char *image)
{
    struct stat statbuf;
    struct gpgpu_image module;
    unsigned char *data;
    size_t done = 0;
    int fd = open(image, O_RDONLY | O_CLOEXEC);
    int result;

    if (fd < 0) {
        return -errno;
    }
    if (fstat(fd, &statbuf) || statbuf.st_size <= 0 ||
        statbuf.st_size > GPGPU_CODE_SIZE) {
        close(fd);
        return -EINVAL;
    }
    data = malloc(statbuf.st_size);
    if (!data) {
        close(fd);
        return -ENOMEM;
    }
    while (done < (size_t)statbuf.st_size) {
        ssize_t count = read(fd, data + done, statbuf.st_size - done);

        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            result = count < 0 ? -errno : -EIO;
            goto out;
        }
        done += count;
    }
    module = (struct gpgpu_image) {
        .host_addr = (uintptr_t)data, .size = done,
    };
    result = request(device, GPGPU_IOCTL_IMAGE, &module);
out:
    free(data);
    close(fd);
    return result;
}

int gpgpuLaunchKernel(gpgpuDevice *device, gpgpuDim3 grid, gpgpuDim3 block,
                      const void *args, size_t args_size)
{
    struct gpgpu_launch launch = {
        .grid = {grid.x, grid.y, grid.z},
        .block = {block.x, block.y, block.z},
        .args = (uintptr_t)args,
        .args_size = args_size,
    };

    if (args_size > 4096 || (args_size && !args)) {
        return -EINVAL;
    }
    return request(device, GPGPU_IOCTL_LAUNCH, &launch);
}

int gpgpuDeviceSynchronize(gpgpuDevice *device)
{
    return device ? 0 : -EINVAL; /* This ABI completes launches synchronously. */
}

void *gpgpuMap(gpgpuDevice *device, gpgpuDevicePtr pointer, size_t size)
{
    if (!device) {
        errno = EINVAL;
        return MAP_FAILED;
    }
    return mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                device->fd, pointer);
}
