# Educational GPGPU SDK

This SDK runs compiled RV32 C++ kernels through a real Linux PCI driver and
`libgpgpu`. It implements advanced experiment 1. The driver uses a coherent DMA
staging buffer, completion interrupts, a page-granular VRAM allocator, and an
allocation-checked BAR2 `mmap` interface. It supports one device and one open
client. Kernels are trusted code; this interface is not a GPU sandbox.

## Build and run

Configure QEMU for `riscv64-softmmu`; Rust is not required for this experiment.
Build `qemu-system-riscv64` and `tests/qtest/qos-test`.

Provide a RISC-V Linux cross compiler, the RISC-V bare-metal multilib compiler,
and a configured Linux kernel build with PCI, `PCI_HOST_GENERIC`, modules,
initramfs and devtmpfs enabled. Build both the Linux `Image` and `modules` so
that `Module.symvers` is available. Linux 6.6 was used for validation.

```sh
make -C contrib/gpgpu CROSS_COMPILE=riscv64-linux-gnu-
make -C contrib/gpgpu driver CROSS_COMPILE=riscv64-linux-gnu- \
    KERNEL_BUILD=/absolute/path/to/linux-build
python3 contrib/gpgpu/scripts/mkinitramfs.py \
    --sdk build/gpgpu-sdk --busybox /path/to/static-riscv64-busybox \
    --output build/gpgpu-sdk/initramfs.cpio
python3 contrib/gpgpu/scripts/run-linux.py \
    --qemu build/qemu-system-riscv64 --kernel /path/to/linux-build/arch/riscv/boot/Image \
    --initrd build/gpgpu-sdk/initramfs.cpio \
    --bios pc-bios/opensbi-riscv64-generic-fw_dynamic.bin \
    --log build/gpgpu-sdk/linux-demo.log
```

The guest loads `gpgpu_pci.ko`, opens `/dev/gpgpu0`, and verifies vector addition,
ReLU, 8x8 matrix multiplication and mapped result reads. All kernel inputs and
outputs pass through the driver DMA path. The host executable is RV64 Linux;
the device code is RV32IF+Zicsr with the ILP32F ABI.

## Programming and launch ABI

`gpgpuOpen`, `gpgpuMalloc`, `gpgpuFree`, `gpgpuMemcpy`, `gpgpuLoadKernel`,
`gpgpuLaunchKernel` and `gpgpuDeviceSynchronize` return zero or a negative errno.
Launches complete synchronously. `gpgpuMap` follows the POSIX `mmap` contract;
its device offset and length must be page-aligned allocation ranges.

Each image starts at VRAM `0x10000` and defines
`extern "C" void gpgpu_kernel(const Args *)`. `device/start.S` calls this
function and executes `ebreak` on return. Before entry the device initializes
`a0` to the argument block, `ra` to zero, and `sp` to the top of a distinct
4 KiB stack for each logical thread. The driver marshals arguments byte-for-byte;
use fixed-width types, including **32-bit device pointers**, rather than copying
host pointers or host-native structures. The reserved code region is 960 KiB;
data, argument and stack allocations begin at 1 MiB.

The SDK adds BAR0 `STACK_ADDR` at `0x334` and `STACK_STRIDE` at `0x338`.
Zero stride preserves the original stack-free test ABI. The device validates
the entire stack arena against VRAM. These registers are SDK extensions, not
changes to the original 17 tests.

`device/kernel.h` supplies `__global__`, `threadIdx`, `blockIdx`, `blockDim`
and `gridDim`. This is a small C++ convention, not a CUDA parser: launch syntax
is the explicit runtime API, and advanced CUDA facilities are not advertised.
Device globals requiring dynamic constructors, exceptions, recursion beyond
the stack limit, shared memory, and cross-warp barriers are outside this ABI.

## Validation

The original 17 QTests remain unchanged. The additional
`tests/qtest/gpgpu-extra-test.c` tests verify real DMA, MSI-X mask delivery,
reset, control flow across blocks/warps, floating-point state and low-precision
edge cases. Run the entire device group recursively:

```sh
QTEST_QEMU_BINARY=build/qemu-system-riscv64 build/tests/qtest/qos-test \
    -r /riscv64/virt/generic-pcihost/pci-bus-generic/pci-bus/gpgpu/gpgpu-tests
```
