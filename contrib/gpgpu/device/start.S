/* SPDX-License-Identifier: MIT */
.section .text.entry,"ax",@progbits
.global _start
_start:
    /* The device sets a0=argument address and sp=private stack top. */
    call gpgpu_kernel
    ebreak
1:  j 1b
