/* SPDX-License-Identifier: GPL-2.0-or-later */
#include "qemu/osdep.h"
#include "qapi/error.h"
#include "hw/pci/pci.h"
#include "gpgpu.h"
#ifdef __linux__
#include <dlfcn.h>
#endif

#ifdef __linux__
static int gpgpu_simx_dma(void *opaque, uint64_t addr, void *buf,
                          uint64_t size, int write)
{
    GPGPUState *s = opaque;
    MemTxResult result = write ? pci_dma_write(PCI_DEVICE(s), addr, buf, size) :
                                pci_dma_read(PCI_DEVICE(s), addr, buf, size);
    return result != MEMTX_OK;
}
#endif

bool gpgpu_simx_init(GPGPUState *s, Error **errp)
{
    if (!s->backend || !strcmp(s->backend, "interpreter")) {
        return true;
    }
    if (strcmp(s->backend, "simx")) {
        error_setg(errp, "GPGPU: unknown backend '%s'", s->backend);
        return false;
    }
#ifdef __linux__
    const GPGPUVortexAPI *(*get_api)(void);

    if (!s->simx_library) {
        error_setg(errp, "GPGPU: backend=simx requires simx_library");
        return false;
    }
    /* Berkeley SoftFloat must remain private from QEMU's SoftFloat ABI. */
    s->simx_module = dlopen(s->simx_library,
                           RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!s->simx_module) {
        error_setg(errp, "GPGPU: loading simx: %s", dlerror());
        return false;
    }
    get_api = dlsym(s->simx_module, "gpgpu_vortex_get_api");
    s->simx_api = get_api ? get_api() : NULL;
    if (!s->simx_api || s->simx_api->abi != GPGPU_VORTEX_ABI ||
        s->simx_api->cores != s->num_cus ||
        s->simx_api->warps != s->warps_per_cu ||
        s->simx_api->lanes != s->warp_size) {
        error_setg(errp, "GPGPU: simx ABI/configuration mismatch");
        gpgpu_simx_fini(s);
        return false;
    }
    s->simx = s->simx_api->create(s->vram_ptr, s->vram_size, s, gpgpu_simx_dma);
    if (!s->simx) {
        error_setg(errp, "GPGPU: simx initialization failed (one instance supported)");
        gpgpu_simx_fini(s);
        return false;
    }
    return true;
#else
    error_setg(errp, "GPGPU: simx backend requires a Linux host");
    return false;
#endif
}

void gpgpu_simx_fini(GPGPUState *s)
{
#ifdef __linux__
    if (s->simx) {
        s->simx_api->destroy(s->simx);
    }
    if (s->simx_module) {
        dlclose(s->simx_module);
    }
#endif
    s->simx = NULL;
    s->simx_module = NULL;
    s->simx_api = NULL;
}
