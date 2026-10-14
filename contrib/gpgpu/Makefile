SDK_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR ?= $(SDK_DIR)/../../build/gpgpu-sdk
BUILD_DIR := $(abspath $(BUILD_DIR))
CROSS_COMPILE ?= riscv64-linux-gnu-
HOST_CC ?= $(CROSS_COMPILE)gcc
DEVICE_CC ?= riscv64-unknown-elf-gcc
DEVICE_OBJCOPY ?= riscv64-unknown-elf-objcopy
DEVICE_CFLAGS := -march=rv32if_zicsr -mabi=ilp32f -O2 -g \
    -ffreestanding -fno-builtin -fno-exceptions -fno-rtti \
    -fno-unwind-tables -fno-asynchronous-unwind-tables \
    -msmall-data-limit=0 -mno-relax -I$(SDK_DIR)/device
KERNELS := vecadd relu matmul

.PHONY: all kernels driver clean
all: $(BUILD_DIR)/gpgpu-demo kernels
kernels: $(addprefix $(BUILD_DIR)/kernels/,$(addsuffix .bin,$(KERNELS)))

$(BUILD_DIR)/gpgpu-demo: $(SDK_DIR)/examples/demo.c $(SDK_DIR)/lib/gpgpu.c \
                       $(SDK_DIR)/include/gpgpu.h $(SDK_DIR)/include/gpgpu_ioctl.h
	mkdir -p $(@D)
	$(HOST_CC) -std=gnu11 -O2 -g -Wall -Wextra -Werror -static \
        -I$(SDK_DIR)/include $(SDK_DIR)/examples/demo.c \
        $(SDK_DIR)/lib/gpgpu.c -o $@

$(BUILD_DIR)/kernels/%.elf: $(SDK_DIR)/device/%.cpp \
                         $(SDK_DIR)/device/start.S $(SDK_DIR)/device/kernel.ld \
                         $(SDK_DIR)/device/kernel.h
	mkdir -p $(@D)
	$(DEVICE_CC) $(DEVICE_CFLAGS) -nostdlib -Wl,--no-relax \
        -Wl,-T,$(SDK_DIR)/device/kernel.ld $(SDK_DIR)/device/start.S $< \
        -lgcc -o $@

$(BUILD_DIR)/kernels/%.bin: $(BUILD_DIR)/kernels/%.elf
	$(DEVICE_OBJCOPY) -O binary $< $@

driver:
	test -n "$(KERNEL_BUILD)"
	mkdir -p $(BUILD_DIR)/driver
	cp $(SDK_DIR)/driver/Makefile $(SDK_DIR)/driver/gpgpu_pci.c \
        $(BUILD_DIR)/driver/
	mkdir -p $(BUILD_DIR)/include
	cp $(SDK_DIR)/include/gpgpu_ioctl.h $(BUILD_DIR)/include/
	$(MAKE) -C $(KERNEL_BUILD) M=$(BUILD_DIR)/driver ARCH=riscv \
        CROSS_COMPILE=$(CROSS_COMPILE) modules

clean:
	rm -rf $(BUILD_DIR)/kernels $(BUILD_DIR)/driver $(BUILD_DIR)/gpgpu-demo

.SECONDARY:
