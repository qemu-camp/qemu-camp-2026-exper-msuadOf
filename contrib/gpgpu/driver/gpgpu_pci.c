// SPDX-License-Identifier: GPL-2.0-or-later
#include <linux/completion.h>
#include <linux/dma-mapping.h>
#include <linux/fs.h>
#include <linux/genalloc.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/kref.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include "gpgpu_ioctl.h"

#define R_CTRL          0x100
#define R_ERROR         0x108
#define R_IRQ_ENABLE    0x200
#define R_IRQ_STATUS    0x204
#define R_IRQ_ACK       0x208
#define R_KERNEL        0x300
#define R_ARGS          0x308
#define R_GRID          0x310
#define R_BLOCK         0x31c
#define R_DISPATCH      0x330
#define R_STACK         0x334
#define R_STACK_STRIDE  0x338
#define R_DMA_SRC       0x400
#define R_DMA_DST       0x408
#define R_DMA_SIZE      0x410
#define R_DMA_CTRL      0x414
#define R_DMA_STATUS    0x418
#define STAGING_SIZE    4096

struct gpgpu_allocation {
    struct list_head node;
    unsigned long addr, size;
};

struct gpgpu_pci {
    struct pci_dev *pdev;
    struct miscdevice misc;
    struct mutex lock;
    struct kref refs;
    struct gen_pool *pool;
    struct completion dma_done, kernel_done;
    void __iomem *regs;
    void __iomem *vram;
    void *staging;
    dma_addr_t staging_dma;
    u32 vram_size;
    int vectors;
    atomic_t opened;
    bool removed;
    struct address_space *mapping;
};

struct gpgpu_file {
    struct gpgpu_pci *gpu;
    struct list_head allocations;
    u32 image_size;
};

static void gpgpu_destroy(struct kref *refs)
{
    struct gpgpu_pci *gpu = container_of(refs, struct gpgpu_pci, refs);

    pci_dev_put(gpu->pdev);
    kfree(gpu);
}

static irqreturn_t gpgpu_irq(int irq, void *opaque)
{
    struct gpgpu_pci *gpu = opaque;
    u32 pending = readl(gpu->regs + R_IRQ_STATUS);

    if (!pending)
        return IRQ_NONE;
    writel(pending, gpu->regs + R_IRQ_ACK);
    if (pending & (1 | 4))
        complete(&gpu->kernel_done);
    if (pending & (2 | 4))
        complete(&gpu->dma_done);
    return IRQ_HANDLED;
}

static void write_addr(struct gpgpu_pci *gpu, unsigned reg, u64 addr)
{
    writel(lower_32_bits(addr), gpu->regs + reg);
    writel(upper_32_bits(addr), gpu->regs + reg + 4);
}

static int wait_done(struct gpgpu_pci *gpu, struct completion *done)
{
    if (!wait_for_completion_timeout(done, msecs_to_jiffies(10000))) {
        writel(2, gpu->regs + R_CTRL);
        writel(7, gpu->regs + R_IRQ_ENABLE);
        writel(1, gpu->regs + R_CTRL);
        return -ETIMEDOUT;
    }
    return readl(gpu->regs + R_ERROR) ? -EIO : 0;
}

static int gpu_copy(struct gpgpu_pci *gpu, u64 device_addr,
                    void __user *host, u32 size, bool from_device)
{
    while (size) {
        u32 chunk = min_t(u32, size, STAGING_SIZE);
        int err;

        if (!from_device && copy_from_user(gpu->staging, host, chunk))
            return -EFAULT;
        reinit_completion(&gpu->dma_done);
        writel(15, gpu->regs + R_ERROR);
        write_addr(gpu, R_DMA_SRC, from_device ? device_addr : gpu->staging_dma);
        write_addr(gpu, R_DMA_DST, from_device ? gpu->staging_dma : device_addr);
        writel(chunk, gpu->regs + R_DMA_SIZE);
        dma_wmb();
        writel(1 | 4 | (from_device ? 2 : 0), gpu->regs + R_DMA_CTRL);
        err = wait_done(gpu, &gpu->dma_done);
        if (err)
            return err;
        if (readl(gpu->regs + R_DMA_STATUS) != 2)
            return -EIO;
        dma_rmb();
        if (from_device && copy_to_user(host, gpu->staging, chunk))
            return -EFAULT;
        size -= chunk;
        device_addr += chunk;
        host = (char __user *)host + chunk;
    }
    return 0;
}

static bool owns_range(struct gpgpu_file *ctx, u64 addr, u64 size)
{
    struct gpgpu_allocation *allocation;

    list_for_each_entry(allocation, &ctx->allocations, node) {
        if (addr >= allocation->addr &&
            addr - allocation->addr <= allocation->size &&
            size <= allocation->size - (addr - allocation->addr))
            return true;
    }
    return false;
}

static int launch_kernel(struct gpgpu_file *ctx, struct gpgpu_launch *launch)
{
    struct gpgpu_pci *gpu = ctx->gpu;
    unsigned long args, stacks, stack_size;
    u64 blocks = 1, threads = 1;
    int err;
    unsigned i;

    if (!ctx->image_size || launch->reserved || launch->args_size > 4096)
        return -EINVAL;
    for (i = 0; i < 3; i++) {
        if (!launch->grid[i] || !launch->block[i] ||
            launch->grid[i] > 524288 / blocks ||
            launch->block[i] > 8192 / threads)
            return -EINVAL;
        blocks *= launch->grid[i];
        threads *= launch->block[i];
    }
    if (blocks * threads > gpu->vram_size / GPGPU_STACK_STRIDE)
        return -ENOMEM;
    stack_size = blocks * threads * GPGPU_STACK_STRIDE;
    args = gen_pool_alloc(gpu->pool, 4096);
    stacks = gen_pool_alloc(gpu->pool, stack_size);
    if (!args || !stacks) {
        err = -ENOMEM;
        goto out;
    }
    err = gpu_copy(gpu, args, u64_to_user_ptr(launch->args),
                   launch->args_size, false);
    if (err)
        goto out;
    reinit_completion(&gpu->kernel_done);
    writel(15, gpu->regs + R_ERROR);
    write_addr(gpu, R_KERNEL, GPGPU_CODE_ADDR);
    write_addr(gpu, R_ARGS, args);
    for (i = 0; i < 3; i++) {
        writel(launch->grid[i], gpu->regs + R_GRID + i * 4);
        writel(launch->block[i], gpu->regs + R_BLOCK + i * 4);
    }
    writel(stacks, gpu->regs + R_STACK);
    writel(GPGPU_STACK_STRIDE, gpu->regs + R_STACK_STRIDE);
    writel(1, gpu->regs + R_DISPATCH);
    err = wait_done(gpu, &gpu->kernel_done);
out:
    if (stacks)
        gen_pool_free(gpu->pool, stacks, stack_size);
    if (args)
        gen_pool_free(gpu->pool, args, 4096);
    return err;
}

static long gpgpu_ioctl(struct file *file, unsigned int command,
                         unsigned long arg)
{
    struct gpgpu_file *ctx = file->private_data;
    struct gpgpu_pci *gpu = ctx->gpu;
    void __user *user = (void __user *)arg;
    struct gpgpu_allocation *allocation, *tmp;
    int err = 0;

    if (mutex_lock_interruptible(&gpu->lock))
        return -ERESTARTSYS;
    if (gpu->removed) {
        err = -ENODEV;
        goto out;
    }
    switch (command) {
    case GPGPU_IOCTL_INFO: {
        struct gpgpu_info info = {
            .abi_version = GPGPU_ABI_VERSION,
            .vram_size = gpu->vram_size,
            .caps = readl(gpu->regs + 8),
        };

        if (copy_to_user(user, &info, sizeof(info)))
            err = -EFAULT;
        break;
    }
    case GPGPU_IOCTL_ALLOC: {
        struct gpgpu_buffer buffer;

        if (copy_from_user(&buffer, user, sizeof(buffer))) {
            err = -EFAULT;
            break;
        }
        if (!buffer.size || buffer.size > gpu->vram_size) {
            err = -EINVAL;
            break;
        }
        allocation = kzalloc(sizeof(*allocation), GFP_KERNEL);
        if (!allocation) {
            err = -ENOMEM;
            break;
        }
        allocation->size = PAGE_ALIGN(buffer.size);
        allocation->addr = gen_pool_alloc(gpu->pool, allocation->size);
        buffer.device_addr = allocation->addr;
        if (!allocation->addr) {
            kfree(allocation);
            err = -ENOMEM;
            break;
        }
        memset_io(gpu->vram + allocation->addr, 0, allocation->size);
        if (copy_to_user(user, &buffer, sizeof(buffer))) {
            gen_pool_free(gpu->pool, allocation->addr, allocation->size);
            kfree(allocation);
            err = -EFAULT;
            break;
        }
        list_add(&allocation->node, &ctx->allocations);
        break;
    }
    case GPGPU_IOCTL_FREE: {
        struct gpgpu_buffer buffer;

        if (copy_from_user(&buffer, user, sizeof(buffer))) {
            err = -EFAULT;
            break;
        }
        err = -ENOENT;
        list_for_each_entry_safe(allocation, tmp, &ctx->allocations, node) {
            if (allocation->addr == buffer.device_addr) {
                gen_pool_free(gpu->pool, allocation->addr, allocation->size);
                list_del(&allocation->node);
                kfree(allocation);
                err = 0;
                break;
            }
        }
        break;
    }
    case GPGPU_IOCTL_COPY: {
        struct gpgpu_copy copy;

        if (copy_from_user(&copy, user, sizeof(copy))) {
            err = -EFAULT;
            break;
        }
        if (copy.from_device > 1 ||
            !owns_range(ctx, copy.device_addr, copy.size)) {
            err = -EINVAL;
            break;
        }
        err = gpu_copy(gpu, copy.device_addr, u64_to_user_ptr(copy.host_addr),
                       copy.size, copy.from_device);
        break;
    }
    case GPGPU_IOCTL_IMAGE: {
        struct gpgpu_image image;

        if (copy_from_user(&image, user, sizeof(image))) {
            err = -EFAULT;
            break;
        }
        if (!image.size || image.size > GPGPU_CODE_SIZE || image.reserved) {
            err = -EINVAL;
            break;
        }
        ctx->image_size = 0;
        memset_io(gpu->vram + GPGPU_CODE_ADDR, 0, GPGPU_CODE_SIZE);
        err = gpu_copy(gpu, GPGPU_CODE_ADDR, u64_to_user_ptr(image.host_addr),
                       image.size, false);
        if (!err)
            ctx->image_size = image.size;
        break;
    }
    case GPGPU_IOCTL_LAUNCH: {
        struct gpgpu_launch launch;

        if (copy_from_user(&launch, user, sizeof(launch))) {
            err = -EFAULT;
            break;
        }
        err = launch_kernel(ctx, &launch);
        break;
    }
    default:
        err = -ENOTTY;
    }
out:
    mutex_unlock(&gpu->lock);
    return err;
}

static int gpgpu_mmap(struct file *file, struct vm_area_struct *vma)
{
    struct gpgpu_file *ctx = file->private_data;
    u64 offset = (u64)vma->vm_pgoff << PAGE_SHIFT;
    unsigned long size = vma->vm_end - vma->vm_start;
    int err = -EINVAL;

    mutex_lock(&ctx->gpu->lock);
    if (!ctx->gpu->removed && owns_range(ctx, offset, size)) {
        vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
        vm_flags_set(vma, VM_IO | VM_DONTEXPAND | VM_DONTDUMP);
        err = remap_pfn_range(vma, vma->vm_start,
            (pci_resource_start(ctx->gpu->pdev, 2) + offset) >> PAGE_SHIFT,
            size, vma->vm_page_prot);
    }
    mutex_unlock(&ctx->gpu->lock);
    return err;
}

static int gpgpu_open(struct inode *inode, struct file *file)
{
    struct gpgpu_pci *gpu = container_of(file->private_data,
                                        struct gpgpu_pci, misc);
    struct gpgpu_file *ctx;

    if (atomic_cmpxchg(&gpu->opened, 0, 1))
        return -EBUSY;
    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx) {
        atomic_set(&gpu->opened, 0);
        return -ENOMEM;
    }
    ctx->gpu = gpu;
    INIT_LIST_HEAD(&ctx->allocations);
    file->private_data = ctx;
    mutex_lock(&gpu->lock);
    if (gpu->removed) {
        mutex_unlock(&gpu->lock);
        atomic_set(&gpu->opened, 0);
        kfree(ctx);
        return -ENODEV;
    }
    kref_get(&gpu->refs);
    gpu->mapping = file->f_mapping;
    writel(2, gpu->regs + R_CTRL);
    writel(7, gpu->regs + R_IRQ_ENABLE);
    writel(1, gpu->regs + R_CTRL);
    mutex_unlock(&gpu->lock);
    return 0;
}

static int gpgpu_release(struct inode *inode, struct file *file)
{
    struct gpgpu_file *ctx = file->private_data;
    struct gpgpu_allocation *allocation, *tmp;

    mutex_lock(&ctx->gpu->lock);
    if (!ctx->gpu->removed)
        writel(2, ctx->gpu->regs + R_CTRL);
    list_for_each_entry_safe(allocation, tmp, &ctx->allocations, node) {
        if (!ctx->gpu->removed)
            gen_pool_free(ctx->gpu->pool, allocation->addr, allocation->size);
        list_del(&allocation->node);
        kfree(allocation);
    }
    atomic_set(&ctx->gpu->opened, 0);
    ctx->gpu->mapping = NULL;
    mutex_unlock(&ctx->gpu->lock);
    kref_put(&ctx->gpu->refs, gpgpu_destroy);
    kfree(ctx);
    return 0;
}

static const struct file_operations gpgpu_fops = {
    .owner = THIS_MODULE,
    .open = gpgpu_open,
    .release = gpgpu_release,
    .unlocked_ioctl = gpgpu_ioctl,
    .mmap = gpgpu_mmap,
    .llseek = no_llseek,
};

static int gpgpu_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    struct gpgpu_pci *gpu;
    int err, i;

    gpu = kzalloc(sizeof(*gpu), GFP_KERNEL);
    if (!gpu)
        return -ENOMEM;
    gpu->pdev = pci_dev_get(pdev);
    kref_init(&gpu->refs);
    err = pcim_enable_device(pdev);
    if (err)
        goto free_gpu;
    err = pcim_iomap_regions(pdev, BIT(0) | BIT(2), "gpgpu");
    if (err)
        goto free_gpu;
    gpu->regs = pcim_iomap_table(pdev)[0];
    gpu->vram = pcim_iomap_table(pdev)[2];
    if (readl(gpu->regs) != 0x47505055) {
        err = -ENODEV;
        goto free_gpu;
    }
    gpu->vram_size = readl(gpu->regs + 0x0c);
    if (gpu->vram_size <= 0x100000 || readl(gpu->regs + 0x10)) {
        err = -EINVAL;
        goto free_gpu;
    }
    if (dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64))) {
        err = -EIO;
        goto free_gpu;
    }
    pci_set_master(pdev);
    mutex_init(&gpu->lock);
    init_completion(&gpu->dma_done);
    init_completion(&gpu->kernel_done);
    gpu->pool = devm_gen_pool_create(&pdev->dev, PAGE_SHIFT, -1, "gpgpu-vram");
    if (IS_ERR(gpu->pool)) {
        err = PTR_ERR(gpu->pool);
        goto free_gpu;
    }
    err = gen_pool_add(gpu->pool, 0x100000, gpu->vram_size - 0x100000, -1);
    if (err)
        goto free_gpu;
    gpu->staging = dmam_alloc_coherent(&pdev->dev, STAGING_SIZE,
                                      &gpu->staging_dma, GFP_KERNEL);
    if (!gpu->staging) {
        err = -ENOMEM;
        goto free_gpu;
    }
    gpu->vectors = pci_alloc_irq_vectors(pdev, 1, 3,
        PCI_IRQ_MSIX | PCI_IRQ_MSI | PCI_IRQ_LEGACY);
    if (gpu->vectors < 0) {
        err = gpu->vectors;
        goto free_gpu;
    }
    for (i = 0; i < gpu->vectors; i++) {
        err = devm_request_irq(&pdev->dev, pci_irq_vector(pdev, i), gpgpu_irq,
                               pci_dev_msi_enabled(pdev) ? 0 : IRQF_SHARED,
                               "gpgpu", gpu);
        if (err)
            goto free_vectors;
    }
    gpu->misc = (struct miscdevice) {
        .minor = MISC_DYNAMIC_MINOR,
        .name = "gpgpu0",
        .fops = &gpgpu_fops,
        .parent = &pdev->dev,
    };
    err = misc_register(&gpu->misc);
    if (err)
        goto free_vectors;
    pci_set_drvdata(pdev, gpu);
    dev_info(&pdev->dev, "GPGPU ABI %d, %u MiB VRAM, %d IRQ vector(s)\n",
              GPGPU_ABI_VERSION, gpu->vram_size >> 20, gpu->vectors);
    return 0;
free_vectors:
    while (i-- > 0)
        devm_free_irq(&pdev->dev, pci_irq_vector(pdev, i), gpu);
    pci_free_irq_vectors(pdev);
free_gpu:
    kref_put(&gpu->refs, gpgpu_destroy);
    return err;
}

static void gpgpu_remove(struct pci_dev *pdev)
{
    struct gpgpu_pci *gpu = pci_get_drvdata(pdev);
    int i;

    misc_deregister(&gpu->misc);
    mutex_lock(&gpu->lock);
    gpu->removed = true;
    if (gpu->mapping)
        unmap_mapping_range(gpu->mapping, 0, 0, 1);
    writel(2, gpu->regs + R_CTRL);
    for (i = 0; i < gpu->vectors; i++)
        devm_free_irq(&pdev->dev, pci_irq_vector(pdev, i), gpu);
    pci_free_irq_vectors(pdev);
    mutex_unlock(&gpu->lock);
    kref_put(&gpu->refs, gpgpu_destroy);
}

static const struct pci_device_id gpgpu_ids[] = {
    { PCI_DEVICE(0x1234, 0x1337) },
    { }
};
MODULE_DEVICE_TABLE(pci, gpgpu_ids);

static struct pci_driver gpgpu_driver = {
    .name = "gpgpu",
    .id_table = gpgpu_ids,
    .probe = gpgpu_probe,
    .remove = gpgpu_remove,
};
module_pci_driver(gpgpu_driver);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Educational GPGPU PCI, DMA and kernel launch driver");
