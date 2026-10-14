/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef GPGPU_VORTEX_BRIDGE_H
#define GPGPU_VORTEX_BRIDGE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define GPGPU_VORTEX_ABI 1
typedef struct GPGPUVortexLaunch {
  uint64_t pc, args;
  uint32_t grid[3], block[3], shared_size;
} GPGPUVortexLaunch;
typedef struct GPGPUVortexAPI {
  uint32_t abi, cores, warps, lanes;
  void *(*create)(uint8_t *vram, uint64_t size, void *opaque,
                  int (*dma)(void *, uint64_t, void *, uint64_t, int));
  void (*destroy)(void *);
  int (*launch)(void *, const GPGPUVortexLaunch *);
  /* 0 = drained, 1 = running, negative = fault. */
  int (*step)(void *, uint32_t cycles);
  int (*reset)(void *);
  uint32_t (*cp_read)(void *, uint32_t);
  int (*cp_write)(void *, uint32_t, uint32_t);
  uint64_t (*cycles)(void *);
  uint64_t (*completed)(void *);
  uint64_t (*mem_requests)(void *);
  uint64_t (*divergence)(void *);
} GPGPUVortexAPI;
const GPGPUVortexAPI *gpgpu_vortex_get_api(void);
#ifdef __cplusplus
}
#endif
#endif
