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
        set_float_default_nan_pattern(0x40, &lane->fp_status);
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

static bool core_rounding_mode(GPGPULane *lane, unsigned rm)
{
    static const FloatRoundMode modes[] = {
        float_round_nearest_even, float_round_to_zero,
        float_round_down, float_round_up, float_round_ties_away,
    };

    if (rm == 7) {
        rm = (lane->fcsr >> 5) & 7;
    }
    if (rm >= G_N_ELEMENTS(modes)) {
        return false;
    }
    set_float_rounding_mode(modes[rm], &lane->fp_status);
    return true;
}

static void core_fp_flags(GPGPULane *lane)
{
    unsigned flags = get_float_exception_flags(&lane->fp_status);

    lane->fcsr |= ((flags & float_flag_inexact) ? 1 : 0) |
                  ((flags & float_flag_underflow) ? 2 : 0) |
                  ((flags & float_flag_overflow) ? 4 : 0) |
                  ((flags & float_flag_divbyzero) ? 8 : 0) |
                  ((flags & float_flag_invalid) ? 16 : 0);
}

static uint32_t core_fclass(uint32_t value)
{
    unsigned exponent = (value >> 23) & 255;
    unsigned fraction = value & 0x7fffff;
    bool negative = value >> 31;

    if (exponent == 255) {
        if (fraction) {
            return (fraction & 0x400000) ? 1u << 9 : 1u << 8;
        }
        return negative ? 1u : 1u << 7;
    }
    if (!exponent) {
        if (!fraction) {
            return negative ? 1u << 3 : 1u << 4;
        }
        return negative ? 1u << 2 : 1u << 5;
    }
    return negative ? 1u << 1 : 1u << 6;
}

static float4_e2m1 core_float32_to_e2m1(uint32_t value, float_status *fp)
{
    static const uint32_t values[] = {
        0, 0x3f000000, 0x3f800000, 0x3fc00000,
        0x40000000, 0x40400000, 0x40800000, 0x40c00000,
    };
    static const uint32_t midpoints[] = {
        0x3e800000, 0x3f400000, 0x3fa00000, 0x3fe00000,
        0x40200000, 0x40600000, 0x40a00000,
    };
    uint32_t magnitude = value & 0x7fffffff;
    unsigned sign = value >> 31;
    unsigned lower = 0, result;
    FloatRoundMode rm = get_float_rounding_mode(fp);

    /* Positive FP32 bit patterns sort numerically, including subnormals. */
    if (magnitude >= values[7]) {
        if (magnitude > values[7]) {
            float_raise(magnitude > 0x7f800000 ? float_flag_invalid :
                        float_flag_inexact, fp);
        }
        return (sign << 3) | 7;
    }
    while (magnitude > values[lower + 1]) {
        lower++;
    }
    if (magnitude == values[lower]) {
        return (sign << 3) | lower;
    }
    if (magnitude == values[lower + 1]) {
        return (sign << 3) | (lower + 1);
    }
    float_raise(float_flag_inexact, fp);
    switch (rm) {
    case float_round_to_zero:
        result = lower;
        break;
    case float_round_down:
        result = lower + sign;
        break;
    case float_round_up:
        result = lower + !sign;
        break;
    default:
        result = lower + (magnitude > midpoints[lower] ||
                 (magnitude == midpoints[lower] &&
                  (rm == float_round_ties_away || (lower & 1))));
        break;
    }
    return (sign << 3) | result;
}

static bool core_exec_fp(GPGPULane *lane, uint32_t insn, uint32_t *result,
                          bool *write_gpr)
{
    unsigned rd = (insn >> 7) & 31;
    unsigned rm = (insn >> 12) & 7;
    unsigned rs1 = (insn >> 15) & 31;
    unsigned rs2 = (insn >> 20) & 31;
    unsigned funct7 = insn >> 25;
    uint32_t a = lane->fpr[rs1], b = lane->fpr[rs2];
    float_status *fp = &lane->fp_status;

    *write_gpr = false;
    set_float_exception_flags(0, fp);
    switch (funct7) {
    case 0x00: /* FADD.S */
    case 0x04: /* FSUB.S */
    case 0x08: /* FMUL.S */
    case 0x0c: /* FDIV.S */
    case 0x2c: /* FSQRT.S */
    case 0x60: /* FCVT.W[U].S */
    case 0x68: /* FCVT.S.W[U] */
    case 0x22: /* BF16 conversion extension */
    case 0x24: /* FP8 conversion extension */
    case 0x26: /* FP4 conversion extension */
        if (!core_rounding_mode(lane, rm)) {
            return false;
        }
        break;
    default:
        break;
    }

    switch (funct7) {
    case 0x00:
        *result = float32_add(a, b, fp);
        break;
    case 0x04:
        *result = float32_sub(a, b, fp);
        break;
    case 0x08:
        *result = float32_mul(a, b, fp);
        break;
    case 0x0c:
        *result = float32_div(a, b, fp);
        break;
    case 0x2c:
        if (rs2) {
            return false;
        }
        *result = float32_sqrt(a, fp);
        break;
    case 0x10: /* FSGNJ.S/FSGNJN.S/FSGNJX.S */
        if (rm > 2) {
            return false;
        }
        *result = (a & 0x7fffffff) |
                  ((rm == 0 ? b : rm == 1 ? ~b : a ^ b) & 0x80000000);
        break;
    case 0x14: /* FMIN.S/FMAX.S */
        if (rm > 1) {
            return false;
        }
        *result = rm ? float32_maxnum(a, b, fp) : float32_minnum(a, b, fp);
        break;
    case 0x50: { /* FLE.S/FLT.S/FEQ.S */
        FloatRelation relation;

        if (rm > 2) {
            return false;
        }
        relation = rm == 2 ? float32_compare_quiet(a, b, fp) :
                             float32_compare(a, b, fp);
        *result = rm == 2 ? relation == float_relation_equal :
                  rm == 1 ? relation == float_relation_less :
                  relation == float_relation_less ||
                  relation == float_relation_equal;
        *write_gpr = true;
        break;
    }
    case 0x60:
        if (rs2 > 1) {
            return false;
        }
        *result = rs2 ? float32_to_uint32(a, fp) : float32_to_int32(a, fp);
        *write_gpr = true;
        break;
    case 0x68:
        if (rs2 > 1) {
            return false;
        }
        *result = rs2 ? uint32_to_float32(lane->gpr[rs1], fp) :
                        int32_to_float32(lane->gpr[rs1], fp);
        break;
    case 0x70: /* FMV.X.W/FCLASS.S */
        if (rs2 || rm > 1) {
            return false;
        }
        *result = rm ? core_fclass(a) : a;
        *write_gpr = true;
        break;
    case 0x78: /* FMV.W.X */
        if (rs2 || rm) {
            return false;
        }
        *result = lane->gpr[rs1];
        break;
    case 0x22:
        if (rs2 > 1) {
            return false;
        }
        *result = rs2 ? float32_to_bfloat16(a, fp) :
                        bfloat16_to_float32(a & 0xffff, fp);
        break;
    case 0x24:
        switch (rs2) {
        case 0:
            *result = bfloat16_to_float32(
                float8_e4m3_to_bfloat16(a & 0xff, fp), fp);
            break;
        case 1:
            *result = float32_to_float8_e4m3(a, true, fp);
            break;
        case 2:
            *result = bfloat16_to_float32(
                float8_e5m2_to_bfloat16(a & 0xff, fp), fp);
            break;
        case 3:
            *result = float32_to_float8_e5m2(a, !float32_is_infinity(a), fp);
            break;
        default:
            return false;
        }
        break;
    case 0x26:
        if (rs2 > 1) {
            return false;
        }
        *result = rs2 ? core_float32_to_e2m1(a, fp) :
            bfloat16_to_float32(float8_e4m3_to_bfloat16(
                float4_e2m1_to_float8_e4m3(a & 15, fp), fp), fp);
        break;
    default:
        return false;
    }
    if (!*write_gpr) {
        lane->fpr[rd] = *result;
    }
    core_fp_flags(lane);
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
    case 0x07: /* FLW */
        if (funct3 != 2 || !core_load(s, a + imm, 4, &result)) {
            return false;
        }
        lane->fpr[rd] = result;
        write_rd = false;
        break;
    case 0x27: /* FSW */
        imm = (int32_t)((insn & 0xfe000000) |
                        ((insn >> 7) & 31) << 20) >> 20;
        if (funct3 != 2 || !core_store(s, a + imm, 4, lane->fpr[rs2])) {
            return false;
        }
        write_rd = false;
        break;
    case 0x53:
        if (!core_exec_fp(lane, insn, &result, &write_rd)) {
            return false;
        }
        break;
    case 0x43: /* FMADD.S */
    case 0x47: /* FMSUB.S */
    case 0x4b: /* FNMSUB.S */
    case 0x4f: { /* FNMADD.S */
        unsigned rs3 = insn >> 27;
        int flags = 0;

        if (((insn >> 25) & 3) || !core_rounding_mode(lane, funct3)) {
            return false;
        }
        if (opcode == 0x47 || opcode == 0x4f) {
            flags |= float_muladd_negate_c;
        }
        if (opcode == 0x4b || opcode == 0x4f) {
            flags |= float_muladd_negate_product;
        }
        set_float_exception_flags(0, &lane->fp_status);
        lane->fpr[rd] = float32_muladd(lane->fpr[rs1], lane->fpr[rs2],
                                      lane->fpr[rs3], flags, &lane->fp_status);
        core_fp_flags(lane);
        write_rd = false;
        break;
    }
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
                    for (unsigned lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
                        warp->lanes[lane].gpr[10] = s->kernel.kernel_args;
                        if (s->kernel.stack_stride) {
                            uint64_t thread = linear_block * threads +
                                              base + lane + 1;

                            warp->lanes[lane].gpr[2] = s->kernel.stack_addr +
                                thread * s->kernel.stack_stride;
                        }
                    }
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
