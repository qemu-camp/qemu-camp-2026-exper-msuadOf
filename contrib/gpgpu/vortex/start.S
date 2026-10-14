/* SPDX-License-Identifier: MIT */
#include <VX_types.h>
.section .text.entry,"ax",@progbits
.global _start
_start:
    li sp, VX_MEM_STACK_BASE_ADDR
    csrr t0, VX_CSR_MHARTID
    slli t0, t0, VX_MEM_STACK_LOG2_SIZE
    sub sp, sp, t0
    csrr s11, VX_CSR_CTA_ENTRY
    csrr a0, VX_CSR_MSCRATCH
    jalr ra, s11
    .insn r 0x0b, 7, 0, x0, x0, x0
    .insn r 0x0b, 0, 0, x0, x0, x0
