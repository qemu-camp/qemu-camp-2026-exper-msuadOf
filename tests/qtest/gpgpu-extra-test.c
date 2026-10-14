/*
 * Additional GPGPU regression tests.
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/module.h"
#include "libqtest.h"
#include "libqos/pci.h"
#include "libqos/libqos-malloc.h"

enum {
    CTRL = 0x100,
    STATUS = 0x104,
    ERROR = 0x108,
    IRQ_ENABLE = 0x200,
    IRQ_STATUS = 0x204,
    IRQ_ACK = 0x208,
    KERNEL_ADDR = 0x300,
    GRID_X = 0x310,
    BLOCK_X = 0x31c,
    DISPATCH = 0x330,
    DMA_SRC_LO = 0x400,
    DMA_SRC_HI = 0x404,
    DMA_DST_LO = 0x408,
    DMA_DST_HI = 0x40c,
    DMA_SIZE = 0x410,
    DMA_CTRL = 0x414,
    DMA_STATUS = 0x418,
};

static QPCIDevice *extra_pci_device(void *obj)
{
    QOSGraphObject *graph = obj;
    QPCIDevice *pdev = graph->get_driver(obj, "pci-device");

    qpci_device_enable(pdev);
    return pdev;
}

static void extra_start_dma(QPCIDevice *pdev, QPCIBar ctrl,
                            uint64_t src, uint64_t dst,
                            uint32_t size, uint32_t flags)
{
    qpci_io_writel(pdev, ctrl, DMA_SRC_LO, src);
    qpci_io_writel(pdev, ctrl, DMA_SRC_HI, src >> 32);
    qpci_io_writel(pdev, ctrl, DMA_DST_LO, dst);
    qpci_io_writel(pdev, ctrl, DMA_DST_HI, dst >> 32);
    qpci_io_writel(pdev, ctrl, DMA_SIZE, size);
    qpci_io_writel(pdev, ctrl, DMA_CTRL, flags | 1);
}

static void test_vram_widths_reset(void *obj, void *data,
                                   QGuestAllocator *alloc)
{
    QPCIDevice *pdev = extra_pci_device(obj);
    QPCIBar ctrl = qpci_iomap(pdev, 0, NULL);
    QPCIBar vram = qpci_iomap(pdev, 2, NULL);

    qpci_io_writeq(pdev, vram, 3, UINT64_C(0x8877665544332211));
    g_assert_cmphex(qpci_io_readb(pdev, vram, 3), ==, 0x11);
    g_assert_cmphex(qpci_io_readw(pdev, vram, 4), ==, 0x3322);
    g_assert_cmphex(qpci_io_readl(pdev, vram, 7), ==, 0x88776655);
    qpci_io_writel(pdev, ctrl, CTRL, 2);
    g_assert_cmphex(qpci_io_readq(pdev, vram, 3), ==,
                    UINT64_C(0x8877665544332211));
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, CTRL), ==, 0);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, STATUS), ==, 1);
}

static void test_dma_roundtrip(void *obj, void *data, QGuestAllocator *alloc)
{
    QPCIDevice *pdev = extra_pci_device(obj);
    QPCIBar ctrl = qpci_iomap(pdev, 0, NULL);
    QPCIBar vram = qpci_iomap(pdev, 2, NULL);
    uint64_t host = guest_alloc(alloc, 64);
    uint8_t input[64], output[64];

    for (unsigned i = 0; i < sizeof(input); i++) {
        input[i] = i ^ 0xa5;
    }
    qtest_memwrite(pdev->bus->qts, host, input, sizeof(input));
    extra_start_dma(pdev, ctrl, host, 0x123, sizeof(input), 4);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, DMA_STATUS), ==, 1);
    qpci_memread(pdev, vram, 0x123, output, sizeof(output));
    g_assert_cmpmem(input, sizeof(input), output, sizeof(output));
    qtest_clock_step(pdev->bus->qts, 1000000);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, DMA_STATUS), ==, 2);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, IRQ_STATUS), ==, 2);
    qpci_io_writel(pdev, ctrl, IRQ_ACK, 2);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, IRQ_STATUS), ==, 0);

    qtest_memset(pdev->bus->qts, host, 0, sizeof(input));
    extra_start_dma(pdev, ctrl, 0x123, host, sizeof(input), 2);
    qtest_clock_step(pdev->bus->qts, 1000000);
    qtest_memread(pdev->bus->qts, host, output, sizeof(output));
    g_assert_cmpmem(input, sizeof(input), output, sizeof(output));
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, IRQ_STATUS), ==, 0);
    guest_free(alloc, host);
}

static void test_dma_fault_reset(void *obj, void *data, QGuestAllocator *alloc)
{
    QPCIDevice *pdev = extra_pci_device(obj);
    QPCIBar ctrl = qpci_iomap(pdev, 0, NULL);
    uint64_t host = guest_alloc(alloc, 16);

    extra_start_dma(pdev, ctrl, host, UINT64_MAX - 7, 16, 0);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, DMA_STATUS), ==, 4);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, ERROR), ==, 8);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, IRQ_STATUS), ==, 4);
    qpci_io_writel(pdev, ctrl, ERROR, 8);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, ERROR), ==, 0);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, STATUS), ==, 1);

    extra_start_dma(pdev, ctrl, host, 0, 16, 4);
    qpci_io_writel(pdev, ctrl, DMA_SIZE, 64);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, DMA_SIZE), ==, 16);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, ERROR), ==, 1);
    qpci_io_writel(pdev, ctrl, CTRL, 2);
    qtest_clock_step(pdev->bus->qts, 2000000);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, DMA_STATUS), ==, 0);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, IRQ_STATUS), ==, 0);
    guest_free(alloc, host);
}

static void test_dma_msix(void *obj, void *data, QGuestAllocator *alloc)
{
    QPCIDevice *pdev = extra_pci_device(obj);
    uint64_t host = guest_alloc(alloc, 16);
    uint64_t message = guest_alloc(alloc, 4);
    uint64_t vector;
    QPCIBar ctrl;

    qpci_msix_enable(pdev);
    ctrl = pdev->msix_table_bar;
    vector = pdev->msix_table_off + 16; /* DMA vector 1 */
    qpci_io_writel(pdev, ctrl, vector, message);
    qpci_io_writel(pdev, ctrl, vector + 4, message >> 32);
    qpci_io_writel(pdev, ctrl, vector + 8, 0xaabbccdd);
    qpci_io_writel(pdev, ctrl, vector + 12, 1); /* Mask vector */
    qpci_io_writel(pdev, ctrl, IRQ_ENABLE, 2);
    qtest_writel(pdev->bus->qts, message, 0);
    extra_start_dma(pdev, ctrl, host, 0, 16, 4);
    qtest_clock_step(pdev->bus->qts, 1000000);
    g_assert_true(qpci_msix_pending(pdev, 1));
    g_assert_cmphex(qtest_readl(pdev->bus->qts, message), ==, 0);
    qpci_io_writel(pdev, ctrl, vector + 12, 0); /* Deliver pending message */
    g_assert_cmphex(qtest_readl(pdev->bus->qts, message), ==, 0xaabbccdd);
    g_assert_false(qpci_msix_pending(pdev, 1));
    qpci_msix_disable(pdev);
    guest_free(alloc, message);
    guest_free(alloc, host);
}

static uint32_t rv_i(unsigned opcode, unsigned rd, unsigned funct3,
                     unsigned rs1, int imm)
{
    return ((uint32_t)imm & 0xfff) << 20 | rs1 << 15 |
           funct3 << 12 | rd << 7 | opcode;
}

static uint32_t rv_r(unsigned rd, unsigned funct3, unsigned rs1, unsigned rs2)
{
    return rs2 << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | 0x33;
}

static uint32_t rv_b(unsigned funct3, unsigned rs1, unsigned rs2, int imm)
{
    uint32_t u = imm;

    return ((u >> 12) & 1) << 31 | ((u >> 5) & 63) << 25 |
           rs2 << 20 | rs1 << 15 | funct3 << 12 |
           ((u >> 1) & 15) << 8 | ((u >> 11) & 1) << 7 | 0x63;
}

static void extra_upload(QPCIDevice *pdev, QPCIBar vram,
                         const uint32_t *code, unsigned count)
{
    for (unsigned i = 0; i < count; i++) {
        qpci_io_writel(pdev, vram, i * 4, code[i]);
    }
}

static void extra_dimensions(QPCIDevice *pdev, QPCIBar ctrl,
                             const uint32_t grid[3], const uint32_t block[3])
{
    qpci_io_writel(pdev, ctrl, CTRL, 1);
    qpci_io_writel(pdev, ctrl, KERNEL_ADDR, 0);
    for (unsigned i = 0; i < 3; i++) {
        qpci_io_writel(pdev, ctrl, GRID_X + i * 4, grid[i]);
        qpci_io_writel(pdev, ctrl, BLOCK_X + i * 4, block[i]);
    }
}

static void test_kernel_control_flow(void *obj, void *data,
                                      QGuestAllocator *alloc)
{
    QPCIDevice *pdev = extra_pci_device(obj);
    QPCIBar ctrl = qpci_iomap(pdev, 0, NULL);
    QPCIBar vram = qpci_iomap(pdev, 2, NULL);
    const uint32_t grid[3] = {2, 2, 1}, block[3] = {5, 7, 1};
    const uint32_t code[] = {
        rv_i(0x73, 5, 2, 0, 0xf14), /* mhartid */
        rv_i(0x13, 6, 5, 5, 13), /* linear block */
        rv_i(0x13, 7, 5, 5, 5),
        rv_i(0x13, 7, 7, 7, 255), /* warp */
        rv_i(0x13, 7, 1, 7, 5),
        rv_i(0x13, 5, 7, 5, 31), /* lane */
        rv_r(7, 0, 7, 5), /* local thread = warp*32 + lane */
        rv_i(0x13, 8, 1, 6, 5),
        rv_i(0x13, 9, 1, 6, 1),
        rv_r(8, 0, 8, 9),
        rv_r(8, 0, 8, 6), /* block*35 */
        rv_r(8, 0, 8, 7), /* global index */
        rv_i(0x13, 10, 0, 0, 3),
        rv_i(0x13, 11, 0, 0, 0),
        rv_r(11, 0, 11, 10), /* sum loop: 3+2+1 */
        rv_i(0x13, 10, 0, 10, -1),
        rv_b(1, 10, 0, -8),
        rv_i(0x13, 12, 7, 7, 1),
        rv_b(0, 12, 0, 8), /* Divergent paths */
        rv_i(0x13, 11, 0, 11, 10),
        rv_i(0x13, 0, 0, 0, 99), /* x0 must remain zero */
        rv_r(11, 0, 11, 0),
        rv_i(0x13, 8, 1, 8, 4), /* 16 bytes per output */
        0x00001e37, /* lui x28, 1 */
        rv_r(28, 0, 28, 8),
        0x00be2023, /* sw x11, 0(x28) */
        0x80000fb7, /* lui x31, 0x80000: CTRL */
        rv_i(0x03, 13, 2, 31, 0),
        0x00de2223, /* sw threadIdx.x, 4(x28) */
        rv_i(0x03, 13, 2, 31, 4),
        0x00de2423, /* sw threadIdx.y, 8(x28) */
        0x014000ef, /* jal ra, +20: call subroutine below */
        0x01ee2623, /* sw x30, 12(x28) */
        0x00100073, /* ebreak */
        0xffffffff, /* Unreachable: a bad jump must fail */
        0xffffffff,
        rv_i(0x13, 30, 0, 0, 42),
        0x00008067, /* ret to caller */
    };

    extra_upload(pdev, vram, code, G_N_ELEMENTS(code));
    extra_dimensions(pdev, ctrl, grid, block);
    qpci_io_writel(pdev, ctrl, DISPATCH, 1);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, ERROR), ==, 0);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, STATUS), ==, 1);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, IRQ_STATUS), ==, 1);
    for (unsigned i = 0; i < 140; i++) {
        unsigned local = i % 35;

        g_assert_cmpuint(qpci_io_readl(pdev, vram, 0x1000 + i * 16), ==,
                         (local & 1) ? 16 : 6);
        g_assert_cmpuint(qpci_io_readl(pdev, vram, 0x1004 + i * 16), ==,
                         local % 5);
        g_assert_cmpuint(qpci_io_readl(pdev, vram, 0x1008 + i * 16), ==,
                         local / 5);
        g_assert_cmpuint(qpci_io_readl(pdev, vram, 0x100c + i * 16), ==, 42);
    }
}

static void test_kernel_faults(void *obj, void *data, QGuestAllocator *alloc)
{
    QPCIDevice *pdev = extra_pci_device(obj);
    QPCIBar ctrl = qpci_iomap(pdev, 0, NULL);
    QPCIBar vram = qpci_iomap(pdev, 2, NULL);
    const uint32_t one[3] = {1, 1, 1};

    qpci_io_writel(pdev, ctrl, DISPATCH, 1); /* Disabled device */
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, ERROR), ==, 1);
    qpci_io_writel(pdev, ctrl, CTRL, 2);
    extra_dimensions(pdev, ctrl, one, one);
    qpci_io_writel(pdev, vram, 0, 0xffffffff); /* Illegal instruction */
    qpci_io_writel(pdev, ctrl, DISPATCH, 1);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, ERROR), ==, 4);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, STATUS), ==, 5);
    qpci_io_writel(pdev, ctrl, ERROR, 4);
    qpci_io_writel(pdev, vram, 0, 0x0000006f); /* Infinite loop */
    qpci_io_writel(pdev, ctrl, DISPATCH, 1);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, ERROR), ==, 4);
    qpci_io_writel(pdev, ctrl, ERROR, 4);
    qpci_io_writel(pdev, vram, 0, 0x00008067); /* ret with ra=0 exits */
    qpci_io_writel(pdev, ctrl, DISPATCH, 1);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, ERROR), ==, 0);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, STATUS), ==, 1);
    qpci_io_writel(pdev, ctrl, KERNEL_ADDR, UINT32_MAX - 3);
    qpci_io_writel(pdev, ctrl, DISPATCH, 1);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, ERROR), ==, 1);
    qpci_io_writel(pdev, ctrl, ERROR, 1);
    qpci_io_writel(pdev, ctrl, KERNEL_ADDR, 0);
    qpci_io_writel(pdev, vram, 0, rv_i(0x13, 1, 0, 0, -1));
    qpci_io_writel(pdev, vram, 4, rv_i(0x03, 2, 2, 1, 0));
    qpci_io_writel(pdev, ctrl, DISPATCH, 1); /* Invalid core address */
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, ERROR) & 4, ==, 4);
}

static uint32_t rv_fp(unsigned funct7, unsigned rd, unsigned rm,
                      unsigned rs1, unsigned rs2)
{
    return funct7 << 25 | rs2 << 20 | rs1 << 15 |
           rm << 12 | rd << 7 | 0x53;
}

static void test_fp_rounding_flags(void *obj, void *data,
                                   QGuestAllocator *alloc)
{
    QPCIDevice *pdev = extra_pci_device(obj);
    QPCIBar ctrl = qpci_iomap(pdev, 0, NULL);
    QPCIBar vram = qpci_iomap(pdev, 2, NULL);
    const uint32_t one[3] = {1, 1, 1};
    const uint32_t inputs[] = {0x40200000, 0, 0x80000000, 0xc0000000};
    const uint32_t expected[] = {
        2, 1, 3, 9, 0x7f800000, 0x80000000, 0,
        0x40c80000, 0x7fc00000, 128, 16,
    };
    const uint32_t code[] = {
        0x00001e37, /* output = 0x1000 */
        0x00002eb7, /* input = 0x2000 */
        rv_i(0x07, 1, 2, 29, 0), /* f1=2.5 */
        rv_i(0x73, 0, 5, 2, 2), /* frm=RDN */
        rv_fp(0x60, 5, 7, 1, 0), /* fcvt.w.s dynamic -> 2 */
        0x005e2023,
        rv_i(0x73, 6, 2, 0, 1), /* read NX */
        0x006e2223,
        rv_i(0x73, 0, 5, 0, 1), /* clear fflags */
        rv_fp(0x60, 7, 3, 1, 0), /* fcvt.w.s RUP -> 3 */
        0x007e2423,
        rv_i(0x07, 2, 2, 29, 4), /* f2=+0 */
        rv_fp(0x0c, 3, 0, 1, 2), /* 2.5/0 -> +Inf, DZ */
        rv_i(0x73, 9, 2, 0, 1),
        0x009e2623,
        0x003e2827, /* fsw f3,16(x28) */
        rv_i(0x07, 4, 2, 29, 8), /* f4=-0 */
        rv_fp(0x14, 5, 0, 2, 4), /* min(+0,-0) */
        0x005e2a27,
        rv_fp(0x14, 5, 1, 2, 4), /* max(+0,-0) */
        0x005e2c27,
        0x10108343, /* fmadd.s f6,f1,f1,f2 -> 6.25 */
        0x006e2e27,
        rv_i(0x73, 0, 5, 0, 1),
        rv_i(0x07, 7, 2, 29, 12), /* f7=-2 */
        rv_fp(0x2c, 8, 0, 7, 0), /* sqrt(-2) -> canonical NaN */
        0x028e2027,
        rv_fp(0x70, 10, 1, 3, 0), /* classify +Inf */
        0x02ae2223,
        rv_i(0x73, 11, 2, 0, 1), /* NV */
        0x02be2423,
        0x00100073,
    };

    for (unsigned i = 0; i < G_N_ELEMENTS(inputs); i++) {
        qpci_io_writel(pdev, vram, 0x2000 + i * 4, inputs[i]);
    }
    extra_upload(pdev, vram, code, G_N_ELEMENTS(code));
    extra_dimensions(pdev, ctrl, one, one);
    qpci_io_writel(pdev, ctrl, DISPATCH, 1);
    g_assert_cmphex(qpci_io_readl(pdev, ctrl, ERROR), ==, 0);
    for (unsigned i = 0; i < G_N_ELEMENTS(expected); i++) {
        g_assert_cmphex(qpci_io_readl(pdev, vram, 0x1000 + i * 4), ==,
                        expected[i]);
    }
}

static void gpgpu_register_extra_tests(void)
{
    qos_add_test("extra-vram-widths-reset", "gpgpu", test_vram_widths_reset,
                 NULL);
    qos_add_test("extra-dma-roundtrip", "gpgpu", test_dma_roundtrip, NULL);
    qos_add_test("extra-dma-fault-reset", "gpgpu", test_dma_fault_reset, NULL);
    qos_add_test("extra-dma-msix", "gpgpu", test_dma_msix, NULL);
    qos_add_test("extra-kernel-control-flow", "gpgpu",
                 test_kernel_control_flow, NULL);
    qos_add_test("extra-kernel-faults", "gpgpu", test_kernel_faults, NULL);
    qos_add_test("extra-fp-rounding-flags", "gpgpu",
                 test_fp_rounding_flags, NULL);
}

libqos_init(gpgpu_register_extra_tests);
