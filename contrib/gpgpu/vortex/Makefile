ROOT := $(abspath ../../..)
VORTEX_SRC ?= $(ROOT)/build/deps/vortex
VORTEX_BUILD ?= $(ROOT)/build/vortex32
TOOLS ?= $(ROOT)/build/vortex-tools
OUT ?= $(ROOT)/build/gpgpu-vortex
CONFIGS := -DVX_CFG_NUM_CORES=4 -DVX_CFG_NUM_WARPS=4 -DVX_CFG_NUM_THREADS=32
HOST_CXX ?= g++-11
CLANG := $(TOOLS)/llvm-vortex/bin/clang++
INCLUDES := -I$(VORTEX_BUILD)/sw -I$(VORTEX_SRC)/sw/common \
    -I$(VORTEX_SRC)/sim/common -I$(VORTEX_SRC)/sim/simx
DEVICE_FLAGS := --target=riscv32-unknown-elf -march=rv32imf_zicsr -mabi=ilp32f \
    -O2 -ffreestanding -fno-builtin -fno-exceptions -fno-rtti \
    -msmall-data-limit=0 -mno-relax -ffunction-sections \
    -Xclang -target-feature -Xclang +xvortex -D__VORTEX__ \
    -DVX_CFG_XLEN=32 $(CONFIGS) -I$(VORTEX_BUILD)/sw \
    -I$(VORTEX_SRC)/sw/kernel/include
.PHONY: all kernels
all: $(OUT)/libgpgpu-vortex.so $(OUT)/crt.o $(VORTEX_BUILD)/sw/runtime/libvortex-qemu.so kernels
$(OUT)/crt.o: crt.c
	mkdir -p $(@D)
	$(TOOLS)/llvm-vortex/bin/clang $(DEVICE_FLAGS) -c $< -o $@
$(OUT)/libgpgpu-vortex.so: bridge.cpp bridge.h
	mkdir -p $(@D)
	$(HOST_CXX) -shared -fPIC -std=c++17 -O2 -Wall -Wextra -Werror \
        -DVX_CFG_XLEN=32 -DVX_CFG_XLEN_32 $(CONFIGS) $(INCLUDES) $< \
        -L$(VORTEX_BUILD)/sim/simx -lsimx \
        -Wl,-rpath,$(VORTEX_BUILD)/sim/simx -o $@
kernels: $(OUT)/kernels/vecadd.bin $(OUT)/kernels/relu.bin $(OUT)/kernels/matmul.bin $(OUT)/kernels/sync.bin $(OUT)/kernels/local.bin $(OUT)/kernels/sgemm.bin
$(VORTEX_BUILD)/sw/runtime/libvortex-qemu.so: qtest_transport.cpp
	$(HOST_CXX) -shared -fPIC -std=c++17 -O2 -Wall -Wextra -Werror \
        -I$(VORTEX_SRC)/sw/runtime/common $< -o $@
$(OUT)/kernels/local.elf: local.cpp start.S kernel.ld
	mkdir -p $(@D)
	$(CLANG) $(DEVICE_FLAGS) -nostdlib --ld-path=/usr/bin/ld.lld \
        -Wl,-T,kernel.ld start.S local.cpp -o $@
$(OUT)/kernels/sync.elf: sync.cpp start.S kernel.ld
	mkdir -p $(@D)
	$(CLANG) $(DEVICE_FLAGS) -nostdlib --ld-path=/usr/bin/ld.lld \
        -Wl,-T,kernel.ld start.S sync.cpp -o $@
$(OUT)/kernels/sgemm.elf: $(VORTEX_SRC)/tests/regression/sgemm/kernel.cpp start.S kernel.ld
	mkdir -p $(@D)
	$(CLANG) $(DEVICE_FLAGS) -Dkernel_main=gpgpu_kernel -nostdlib --ld-path=/usr/bin/ld.lld \
        -Wl,-T,kernel.ld start.S $< -o $@
$(OUT)/kernels/%.elf: kernels.cpp start.S kernel.ld
	mkdir -p $(@D)
	$(CLANG) $(DEVICE_FLAGS) -D$(shell echo $* | tr a-z A-Z) -nostdlib \
        --ld-path=/usr/bin/ld.lld -Wl,-T,kernel.ld start.S kernels.cpp -o $@
$(OUT)/kernels/%.bin: $(OUT)/kernels/%.elf
	$(TOOLS)/llvm-vortex/bin/llvm-objcopy -O binary $< $@
.SECONDARY:
