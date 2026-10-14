/* SPDX-License-Identifier: GPL-2.0-or-later */
#include "bridge.h"
#include <VX_types.h>
#include <cmd_processor.h>
#include <core.h>
#include <cstdio>
#include <cstring>
#include <mem.h>
#include <memory>
#include <processor.h>
#include <simobject.h>
#include <stdexcept>
#include <vector>

namespace {
/* SimPlatform is process-global in upstream simx. */
bool claimed;
class Vram : public vortex::RAM {
  uint8_t *bytes_;
  uint64_t size_;
  std::vector<uint8_t> stacks_;

public:
  Vram(uint8_t *p, uint64_t n)
      : RAM(0, 4096), bytes_(p), size_(n),
        stacks_(VX_CFG_NUM_CORES * VX_CFG_NUM_WARPS * VX_CFG_NUM_THREADS *
                (1u << VX_MEM_STACK_LOG2_SIZE)) {}
  uint64_t size() const override { return size_; }
  uint8_t *view(uint64_t a, uint64_t n) {
    if (a <= size_ && n <= size_ - a)
      return bytes_ + a;
    uint64_t base = uint64_t(VX_MEM_STACK_BASE_ADDR) - stacks_.size();
    if (a >= base && a - base <= stacks_.size() &&
        n <= stacks_.size() - (a - base))
      return stacks_.data() + a - base;
    char message[100];
    std::snprintf(message, sizeof(message), "VRAM fault at 0x%llx (%llu bytes)",
                  (unsigned long long)a, (unsigned long long)n);
    throw std::runtime_error(message);
  }
  void read(void *d, uint64_t a, uint64_t n) override {
    std::memcpy(d, view(a, n), n);
  }
  void write(const void *d, uint64_t a, uint64_t n) override {
    std::memcpy(view(a, n), d, n);
  }
};
struct Device {
  Vram ram;
  std::unique_ptr<vortex::Processor> processor;
  void *opaque;
  int (*dma)(void *, uint64_t, void *, uint64_t, int);
  bool running = false, fault = false;
  uint64_t launch_cycles = 0;
  uint64_t completions = 0, requests = 0;
  std::unique_ptr<vortex::CommandProcessor> cp;
  Device(uint8_t *p, uint64_t n, void *o,
         int (*d)(void *, uint64_t, void *, uint64_t, int))
      : ram(p, n), processor(new vortex::Processor), opaque(o), dma(d) {
    vortex::SimPlatform::instance().mt_limit(1);
    processor->attach_ram(&ram);
    processor->set_mem_telemetry_hook(
        [this](const vortex::MemReq &) { ++requests; });
    vortex::CommandProcessor::Hooks h;
    h.dram_read = [this](uint64_t a, void *p, size_t n) {
      transfer(a, p, n, 0);
    };
    h.dram_write = [this](uint64_t a, const void *p, size_t n) {
      transfer(a, const_cast<void *>(p), n, 1);
    };
    h.vortex_dcr_write = [this](uint32_t a, uint32_t v) {
      processor->dcr_write(a, v);
    };
    h.vortex_dcr_read = [this](uint32_t a, uint32_t t) {
      uint32_t v = 0;
      processor->dcr_read(a, t, &v);
      return v;
    };
    h.vortex_start = [this]() { start(); };
    h.vortex_busy = [this]() { return running; };
    cp.reset(new vortex::CommandProcessor(h));
  }
  void transfer(uint64_t a, void *p, size_t n, int wr) {
    if (a < ram.size()) {
      if (wr)
        ram.write(p, a, n);
      else
        ram.read(p, a, n);
    } else if (dma(opaque, a, p, n, wr)) {
      throw vortex::BadAddress();
    }
  }
  void start() {
    processor->reset();
    running = true;
    launch_cycles = requests = 0;
  }
};
void *create(uint8_t *p, uint64_t n, void *o,
             int (*d)(void *, uint64_t, void *, uint64_t, int)) {
  if (claimed)
    return nullptr;
  try {
    auto dev = new Device(p, n, o, d);
    claimed = true;
    return dev;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "simx: %s\n", e.what());
    return nullptr;
  }
}
void destroy(void *p) {
  delete static_cast<Device *>(p);
  claimed = false;
}
int launch(void *p, const GPGPUVortexLaunch *l) {
  auto &d = *static_cast<Device *>(p);
  if (d.running || d.fault)
    return -1;
  uint64_t threads = uint64_t(l->block[0]) * l->block[1] * l->block[2];
  if (!threads || threads > VX_CFG_NUM_WARPS * VX_CFG_NUM_THREADS ||
      l->shared_size > (1u << VX_CFG_LMEM_LOG_SIZE))
    return -1;
  auto wr = [&d](uint32_t a, uint32_t v) { d.processor->dcr_write(a, v); };
  wr(VX_DCR_KMU_STARTUP_ADDR0, l->pc);
  wr(VX_DCR_KMU_STARTUP_ADDR1, l->pc >> 32);
  /* SDK images put their C entry immediately after the startup at +64. */
  wr(VX_DCR_KMU_KERNEL_ENTRY0, l->pc + 64);
  wr(VX_DCR_KMU_KERNEL_ENTRY1, 0);
  wr(VX_DCR_KMU_STARTUP_ARG0, l->args);
  wr(VX_DCR_KMU_STARTUP_ARG1, l->args >> 32);
  for (unsigned i = 0; i < 3; ++i) {
    wr(VX_DCR_KMU_BLOCK_DIM_X + i, l->block[i]);
    wr(VX_DCR_KMU_GRID_DIM_X + i, l->grid[i]);
    wr(VX_DCR_KMU_CLUSTER_DIM_X + i, 1);
  }
  wr(VX_DCR_KMU_BLOCK_SIZE, threads);
  wr(VX_DCR_KMU_WARP_STEP_X, VX_CFG_NUM_THREADS % l->block[0]);
  wr(VX_DCR_KMU_WARP_STEP_Y, (VX_CFG_NUM_THREADS / l->block[0]) % l->block[1]);
  wr(VX_DCR_KMU_WARP_STEP_Z,
     (VX_CFG_NUM_THREADS / (l->block[0] * l->block[1])) % l->block[2]);
  wr(VX_DCR_KMU_LMEM_SIZE, l->shared_size);
  d.start();
  return 0;
}
int step(void *p, uint32_t budget) {
  auto &d = *static_cast<Device *>(p);
  if (d.fault)
    return -1;
  try {
    for (uint32_t i = 0; i < budget; ++i) {
      if (d.running) {
        d.running = d.processor->cycle();
        if (!d.running)
          ++d.completions;
        if (++d.launch_cycles > 10000000)
          throw std::runtime_error("cycle limit");
      }
      d.cp->tick();
      if (!d.running && !d.cp->busy())
        return 0;
    }
    return 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "simx execution fault: %s\n", e.what());
    d.fault = true;
    d.running = false;
    return -1;
  }
}
int reset(void *p) {
  auto &d = *static_cast<Device *>(p);
  /* Upstream reset requires quiescence; faults can leave cache transactions. */
  d.processor.reset();
  /* Dropped channel reservations survive finalize in upstream's counter. */
  vortex::SimChannelBase::inflight_count() = 0;
  d.running = false;
  try {
    d.processor.reset(new vortex::Processor());
    vortex::SimPlatform::instance().mt_limit(1);
    d.processor->attach_ram(&d.ram);
    d.processor->set_mem_telemetry_hook(
        [&d](const vortex::MemReq &) { ++d.requests; });
    d.cp->mmio_write(0, 2);
    d.fault = false;
    d.completions = d.requests = 0;
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "simx reset fault: %s\n", e.what());
    d.fault = true;
    return -1;
  }
}
uint32_t cp_read(void *p, uint32_t a) {
  return static_cast<Device *>(p)->cp->mmio_read(a);
}
int cp_write(void *p, uint32_t a, uint32_t v) {
  try {
    static_cast<Device *>(p)->cp->mmio_write(a, v);
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}
uint64_t cycles(void *) { return vortex::SimPlatform::instance().cycles(); }
uint64_t completed(void *p) { return static_cast<Device *>(p)->completions; }
uint64_t requests(void *p) { return static_cast<Device *>(p)->requests; }
uint64_t divergence(void *p) {
  auto &d = *static_cast<Device *>(p);
  return d.processor ? d.processor->get_first_core()->perf_stats().divergence
                     : 0;
}
const GPGPUVortexAPI api = {GPGPU_VORTEX_ABI,
                            VX_CFG_NUM_CORES,
                            VX_CFG_NUM_WARPS,
                            VX_CFG_NUM_THREADS,
                            create,
                            destroy,
                            launch,
                            step,
                            reset,
                            cp_read,
                            cp_write,
                            cycles,
                            completed,
                            requests,
                            divergence};
} // namespace
extern "C" const GPGPUVortexAPI *gpgpu_vortex_get_api() { return &api; }
