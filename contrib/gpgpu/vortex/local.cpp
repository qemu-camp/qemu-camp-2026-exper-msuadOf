/* SPDX-License-Identifier: MIT */
#include <vx_spawn2.h>
struct Args {
  unsigned a, b, c, n;
};
extern "C" __attribute__((section(".text.gpgpu_kernel"),
                          annotate("vortex.kernel"))) void
gpgpu_kernel(Args *__UNIFORM__ p) {
  unsigned i = threadIdx.x;
  volatile unsigned *scratch = (volatile unsigned *)__local_mem();
  unsigned *out = (unsigned *)p->c;
  scratch[i] = i + 1;
  __syncthreads();
  unsigned other = scratch[(i + 32) % p->n];
  if (i & 1)
    out[i] = other * 3;
  else
    out[i] = other + 7;
  __syncthreads();
}
