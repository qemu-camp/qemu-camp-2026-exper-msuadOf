/* SPDX-License-Identifier: MIT */
/* The Vortex startup uses the newlib entry; its prebuilt libc32 is musl. */
typedef void (*initializer)(void);
extern initializer __preinit_array_start[], __preinit_array_end[];
extern initializer __init_array_start[], __init_array_end[];
void __libc_init_array(void) {
  for (initializer *p = __preinit_array_start; p != __preinit_array_end; ++p)
    (*p)();
  for (initializer *p = __init_array_start; p != __init_array_end; ++p)
    (*p)();
}
