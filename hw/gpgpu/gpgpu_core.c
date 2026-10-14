/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/bswap.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));
    warp->thread_id_base = thread_id_base;
    warp->warp_id = warp_id;
    memcpy(warp->block_id, block_id, sizeof(warp->block_id));
    num_threads = MIN(num_threads, GPGPU_WARP_SIZE);
    for (unsigned i = 0; i < num_threads; i++) {
        GPGPULane *lane = &warp->lanes[i];

        lane->pc = pc;
        lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id, i);
        lane->active = true;
        warp->active_mask |= 1u << i;
        set_float_rounding_mode(float_round_nearest_even, &lane->fp_status);
        set_default_nan_mode(true, &lane->fp_status);
    }
}

static bool core_range_valid(GPGPUState *s, uint32_t addr, unsigned size)
{
    if (addr > s->vram_size || size > s->vram_size - addr) {
        gpgpu_set_error(s, GPGPU_ERR_VRAM_FAULT);
        return false;
    }
    return true;
}

static bool core_load(GPGPUState *s, uint32_t addr, unsigned size,
                       uint32_t *value)
{
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        if (size != 4 || (addr & 3)) {
            return false;
        }
        switch (addr) {
        case GPGPU_CORE_CTRL_THREAD_ID_X ... GPGPU_CORE_CTRL_THREAD_ID_Z:
            *value = s->simt.thread_id[
                (addr - GPGPU_CORE_CTRL_THREAD_ID_X) / 4];
            break;
        case GPGPU_CORE_CTRL_BLOCK_ID_X ... GPGPU_CORE_CTRL_BLOCK_ID_Z:
            *value = s->simt.block_id[
                (addr - GPGPU_CORE_CTRL_BLOCK_ID_X) / 4];
            break;
        case GPGPU_CORE_CTRL_BLOCK_DIM_X ... GPGPU_CORE_CTRL_BLOCK_DIM_Z:
            *value = s->kernel.block_dim[
                (addr - GPGPU_CORE_CTRL_BLOCK_DIM_X) / 4];
            break;
        case GPGPU_CORE_CTRL_GRID_DIM_X ... GPGPU_CORE_CTRL_GRID_DIM_Z:
            *value = s->kernel.grid_dim[
                (addr - GPGPU_CORE_CTRL_GRID_DIM_X) / 4];
            break;
        case GPGPU_CORE_CTRL_WARP_ID:
            *value = s->simt.warp_id;
            break;
        case GPGPU_CORE_CTRL_LANE_ID:
            *value = s->simt.lane_id;
            break;
        default:
            return false;
        }
        return true;
    }
    if (!core_range_valid(s, addr, size)) {
        return false;
    }
    *value = 0;
    for (unsigned i = 0; i < size; i++) {
        *value |= (uint32_t)s->vram_ptr[addr + i] << (i * 8);
    }
    return true;
}

static bool core_store(GPGPUState *s, uint32_t addr, unsigned size,
                        uint32_t value)
{
    if (!core_range_valid(s, addr, size)) {
        return false;
    }
    for (unsigned i = 0; i < size; i++) {
        s->vram_ptr[addr + i] = value >> (i * 8);
    }
    return true;
}

static bool core_csr(GPGPULane *lane, uint32_t insn, uint32_t *value)
{
    unsigned csr = insn >> 20;
    unsigned funct3 = (insn >> 12) & 7;
    unsigned rs1 = (insn >> 15) & 31;
    uint32_t operand = (funct3 & 4) ? rs1 : lane->gpr[rs1];
    uint32_t mask;
    unsigned shift;
    bool write = (funct3 & 3) == 1 || rs1 != 0;

    if (!(funct3 & 3)) {
        return false;
    }
    if (csr == CSR_MHARTID) {
        *value = lane->mhartid;
        return !write;
    }
    switch (csr) {
    case CSR_FFLAGS:
        mask = 0x1f;
        shift = 0;
        break;
    case CSR_FRM:
        mask = 7;
        shift = 5;
        break;
    case CSR_FCSR:
        mask = 0xff;
        shift = 0;
        break;
    default:
        return false;
    }
    *value = (lane->fcsr >> shift) & mask;
    if (write) {
        uint32_t next;

        switch (funct3 & 3) {
        case 1:
            next = operand;
            break;
        case 2:
            next = *value | operand;
            break;
        default:
            next = *value & ~operand;
            break;
        }
        lane->fcsr = (lane->fcsr & ~(mask << shift)) |
                     ((next & mask) << shift);
    }
    return true;
}

static bool core_exec_inst(GPGPUState *s, GPGPULane *lane, uint32_t insn)
{
    unsigned opcode = insn & 0x7f;
    unsigned rd = (insn >> 7) & 31;
    unsigned funct3 = (insn >> 12) & 7;
    unsigned rs1 = (insn >> 15) & 31;
    unsigned rs2 = (insn >> 20) & 31;
    unsigned funct7 = insn >> 25;
    uint32_t a = lane->gpr[rs1], b = lane->gpr[rs2];
    uint32_t next_pc = lane->pc + 4, result = 0;
    int32_t imm = (int32_t)insn >> 20;
    bool write_rd = true;

    switch (opcode) {
    case 0x37: /* LUI */
        result = insn & 0xfffff000;
        break;
    case 0x17: /* AUIPC */
        result = lane->pc + (insn & 0xfffff000);
        break;
    case 0x6f: /* JAL */
        imm = (int32_t)((insn & 0x80000000) | ((insn & 0xff000) << 11) |
                        ((insn & 0x100000) << 2) |
                        ((insn & 0x7fe00000) >> 9)) >> 11;
        result = next_pc;
        next_pc = lane->pc + imm;
        break;
    case 0x67: /* JALR; returning to ra=0 exits the kernel */
        if (funct3) {
            return false;
        }
        if (insn == 0x00008067 && !a) {
            lane->active = false;
            return true;
        }
        result = next_pc;
        next_pc = (a + imm) & ~1u;
        break;
    case 0x63: { /* Conditional branches */
        bool take;

        imm = (int32_t)((insn & 0x80000000) | ((insn & 0x80) << 23) |
                        ((insn & 0x7e000000) >> 1) |
                        ((insn & 0xf00) << 12)) >> 19;
        switch (funct3) {
        case 0:
            take = a == b;
            break;
        case 1:
            take = a != b;
            break;
        case 4:
            take = (int32_t)a < (int32_t)b;
            break;
        case 5:
            take = (int32_t)a >= (int32_t)b;
            break;
        case 6:
            take = a < b;
            break;
        case 7:
            take = a >= b;
            break;
        default:
            return false;
        }
        if (take) {
            next_pc = lane->pc + imm;
        }
        write_rd = false;
        break;
    }
    case 0x03: { /* LB/LH/LW/LBU/LHU */
        unsigned size;

        if (funct3 != 0 && funct3 != 1 && funct3 != 2 &&
            funct3 != 4 && funct3 != 5) {
            return false;
        }
        size = 1u << (funct3 & 3);
        if (!core_load(s, a + imm, size, &result)) {
            return false;
        }
        if (funct3 == 0) {
            result = (int8_t)result;
        } else if (funct3 == 1) {
            result = (int16_t)result;
        }
        break;
    }
    case 0x23: /* SB/SH/SW */
        imm = (int32_t)((insn & 0xfe000000) |
                        ((insn >> 7) & 31) << 20) >> 20;
        if (funct3 > 2 || !core_store(s, a + imm, 1u << funct3, b)) {
            return false;
        }
        write_rd = false;
        break;
    case 0x13: /* Integer immediate operations */
        switch (funct3) {
        case 0:
            result = a + imm;
            break;
        case 2:
            result = (int32_t)a < imm;
            break;
        case 3:
            result = a < (uint32_t)imm;
            break;
        case 4:
            result = a ^ imm;
            break;
        case 6:
            result = a | imm;
            break;
        case 7:
            result = a & imm;
            break;
        case 1:
            if (funct7) {
                return false;
            }
            result = a << rs2;
            break;
        case 5:
            if (funct7 == 0) {
                result = a >> rs2;
            } else if (funct7 == 0x20) {
                result = (int32_t)a >> rs2;
            } else {
                return false;
            }
            break;
        }
        break;
    case 0x33: /* Integer register operations */
        if (funct7 != 0 && funct7 != 0x20) {
            return false;
        }
        if (funct7 == 0x20 && funct3 != 0 && funct3 != 5) {
            return false;
        }
        switch (funct3) {
        case 0:
            result = funct7 ? a - b : a + b;
            break;
        case 1:
            result = a << (b & 31);
            break;
        case 2:
            result = (int32_t)a < (int32_t)b;
            break;
        case 3:
            result = a < b;
            break;
        case 4:
            result = a ^ b;
            break;
        case 5:
            result = funct7 ? (uint32_t)((int32_t)a >> (b & 31)) :
                              a >> (b & 31);
            break;
        case 6:
            result = a | b;
            break;
        case 7:
            result = a & b;
            break;
        }
        break;
    case 0x0f: /* FENCE/FENCE.I: memory is synchronous in this model */
        if (funct3 > 1) {
            return false;
        }
        write_rd = false;
        break;
    case 0x73:
        if (insn == 0x00100073) {
            lane->active = false;
            return true;
        }
        if (!core_csr(lane, insn, &result)) {
            return false;
        }
        break;
    default:
        return false;
    }

    if (next_pc & 3) {
        return false;
    }
    if (write_rd && rd) {
        lane->gpr[rd] = result;
    }
    lane->gpr[0] = 0;
    lane->pc = next_pc;
    return true;
}

int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    for (uint32_t cycle = 0; cycle < max_cycles && warp->active_mask; cycle++) {
        for (unsigned i = 0; i < GPGPU_WARP_SIZE; i++) {
            GPGPULane *lane = &warp->lanes[i];
            uint32_t thread = warp->thread_id_base + i;
            uint32_t insn;

            if (!(warp->active_mask & (1u << i))) {
                continue;
            }
            s->simt.thread_id[0] = thread % s->kernel.block_dim[0];
            thread /= s->kernel.block_dim[0];
            s->simt.thread_id[1] = thread % s->kernel.block_dim[1];
            s->simt.thread_id[2] = thread / s->kernel.block_dim[1];
            memcpy(s->simt.block_id, warp->block_id, sizeof(warp->block_id));
            s->simt.warp_id = warp->warp_id;
            s->simt.lane_id = i;
            s->simt.thread_mask = warp->active_mask;

            if ((lane->pc & 3) || !core_range_valid(s, lane->pc, 4)) {
                return -1;
            }
            insn = ldl_le_p(s->vram_ptr + lane->pc);
            if (!core_exec_inst(s, lane, insn)) {
                qemu_log_mask(LOG_GUEST_ERROR,
                              "GPGPU: invalid instruction 0x%08x at 0x%08x\n",
                              insn, lane->pc);
                return -1;
            }
            if (!lane->active) {
                warp->active_mask &= ~(1u << i);
            }
        }
    }
    s->simt.thread_mask = warp->active_mask;
    return warp->active_mask ? -1 : 0;
}

int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint32_t block[3];
    uint32_t linear_block = 0;
    uint64_t threads = 1, blocks = 1;
    g_autofree GPGPUWarp *warp = g_new0(GPGPUWarp, 1);

    for (unsigned i = 0; i < 3; i++) {
        if (!s->kernel.block_dim[i] || !s->kernel.grid_dim[i] ||
            s->kernel.block_dim[i] > 8192 / threads ||
            s->kernel.grid_dim[i] > (1u << MHARTID_BLOCK_BITS) / blocks) {
            return -1;
        }
        threads *= s->kernel.block_dim[i];
        blocks *= s->kernel.grid_dim[i];
    }
    for (block[2] = 0; block[2] < s->kernel.grid_dim[2]; block[2]++) {
        for (block[1] = 0; block[1] < s->kernel.grid_dim[1]; block[1]++) {
            for (block[0] = 0; block[0] < s->kernel.grid_dim[0]; block[0]++) {
                for (uint32_t base = 0; base < threads;
                     base += GPGPU_WARP_SIZE) {
                    gpgpu_core_init_warp(warp, s->kernel.kernel_addr, base,
                                         block,
                                         MIN(threads - base, GPGPU_WARP_SIZE),
                                         base / GPGPU_WARP_SIZE, linear_block);
                    if (gpgpu_core_exec_warp(s, warp, GPGPU_MAX_CYCLES)) {
                        return -1;
                    }
                }
                linear_block++;
            }
        }
    }
    return 0;
}
