/* SPDX-License-Identifier: MIT */
#include <vx_spawn2.h>
struct Args {
  unsigned a, b, c, n;
};
extern "C" __attribute__((section(".text.gpgpu_kernel"),
                          annotate("vortex.kernel"))) void
gpgpu_kernel(Args *__UNIFORM__ p) {
  float *a = (float *)p->a, *b = (float *)p->b, *c = (float *)p->c;
#if defined(VECADD)
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p->n)
    c[i] = a[i] + b[i];
#elif defined(RELU)
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p->n)
    c[i] = a[i] > 0 ? a[i] : 0;
#else
  unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0;
  for (unsigned k = 0; k < p->n; ++k)
    sum += a[row * p->n + k] * b[k * p->n + col];
  c[row * p->n + col] = sum;
#endif
}
