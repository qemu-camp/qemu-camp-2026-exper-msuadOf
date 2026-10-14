/* SPDX-License-Identifier: MIT */
OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS {
    . = 0x10000;
    .startup : { KEEP(*(.text.entry)) }
    ASSERT(SIZEOF(.startup) <= 64, "Startup exceeds ABI entry offset")
    . = 0x10040;
    .kernel : { KEEP(*(.text.gpgpu_kernel)) }
    .text : { *(.text*) }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) *(.sdata*) }
    .bss : { *(.bss*) *(.sbss*) *(COMMON) }
    /DISCARD/ : { *(.eh_frame*) *(.comment*) }
    ASSERT(. < 0x100000, "Kernel exceeds reserved code region")
}
