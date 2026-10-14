/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef GPGPU_IOCTL_H
#define GPGPU_IOCTL_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define GPGPU_ABI_VERSION 1
#define GPGPU_CODE_ADDR 0x10000
#define GPGPU_CODE_SIZE 0xf0000
#define GPGPU_STACK_STRIDE 4096

struct gpgpu_info {
    __u32 abi_version;
    __u32 vram_size;
    __u32 caps;
    __u32 reserved;
};

struct gpgpu_buffer {
    __u64 size;
    __u64 device_addr;
};

struct gpgpu_copy {
    __u64 host_addr;
    __u64 device_addr;
    __u32 size;
    __u32 from_device;
};

struct gpgpu_image {
    __u64 host_addr;
    __u32 size;
    __u32 reserved;
};

struct gpgpu_launch {
    __u32 grid[3];
    __u32 block[3];
    __u64 args;
    __u32 args_size;
    __u32 reserved;
};

#define GPGPU_IOCTL_INFO   _IOR('G', 0, struct gpgpu_info)
#define GPGPU_IOCTL_ALLOC  _IOWR('G', 1, struct gpgpu_buffer)
#define GPGPU_IOCTL_FREE   _IOW('G', 2, struct gpgpu_buffer)
#define GPGPU_IOCTL_COPY   _IOW('G', 3, struct gpgpu_copy)
#define GPGPU_IOCTL_IMAGE  _IOW('G', 4, struct gpgpu_image)
#define GPGPU_IOCTL_LAUNCH _IOW('G', 5, struct gpgpu_launch)

#endif
