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

static void gpgpu_register_extra_tests(void)
{
    qos_add_test("extra-vram-widths-reset", "gpgpu", test_vram_widths_reset,
                 NULL);
    qos_add_test("extra-dma-roundtrip", "gpgpu", test_dma_roundtrip, NULL);
    qos_add_test("extra-dma-fault-reset", "gpgpu", test_dma_fault_reset, NULL);
    qos_add_test("extra-dma-msix", "gpgpu", test_dma_msix, NULL);
}

libqos_init(gpgpu_register_extra_tests);
