# GPGPU 实验记录

范围为基础实验一至十、进阶实验一和二，每个实验单独一个 commit。
未修改 CPU、SoC、Rust 方向的实现。

| 实验 | 主要代码 | 对应验证 | Commit |
|---|---|---|---|
| 一：设备 ID | `hw/gpgpu/gpgpu.c` | device-id | `a7fa443` |
| 二：显存容量 | 同上 | vram-size | `0220556` |
| 三：控制与复位 | 同上 | global-ctrl | `5238078` |
| 四：分发寄存器 | `gpgpu.c`, `gpgpu.h` | dispatch-regs | `70b6869` |
| 五：显存访问 | `gpgpu.c` | vram-access | `7acb5df` |
| 六：DMA 与 IRQ | `gpgpu.c` | dma-regs、irq-regs | `2844e95` |
| 七：SIMT 上下文 | `gpgpu.c`, `gpgpu.h` | 五项上下文/复位测试 | `5a2f26e` |
| 八：RV32I 执行 | `gpgpu_core.c`, `gpgpu_core.h` | kernel-exec | `2fbee71` |
| 九：RV32F 执行 | `gpgpu_core.c`, `gpgpu_core.h` | fp-kernel-exec | `aa3e7b3` |
| 十：低精度转换 | `gpgpu_core.c` | 三项 BF16/FP8/FP4 测试 | `cad627d` |
| 进阶一：类 CUDA 软件栈 | `contrib/gpgpu/{driver,lib,device,examples,scripts}` | Linux 驱动、三个算子、BAR2 mmap | `69bd939` |
| 进阶二：真实 Vortex simx | `gpgpu.c`, `gpgpu.h`, `contrib/gpgpu/vortex/` | 原生/PoCL sgemm、屏障/共享内存/分支、MSI-X、两套 OS | 本记录所在 commit |

原始 `tests/qtest/gpgpu-test.c` 未修改。17 项计分测试以及
`gpgpu-extra-test.c` 中 8 项补充测试全部通过。使用 `-r` 递归运行设备组；
`-p` 必须给出具体测试路径，否则可能选择不到测试。

[进阶一构建、ABI 与边界](README.md)；
[进阶二架构、完整编译链路、复现方法与验证边界](vortex/README.md)。

基础解释器按 lane/warp 顺序执行，缺少硬件流水线的并行时序。
进阶二使用真实 simx 的多 core/warp 调度模型；模拟器在宿主侧单线程推进，
保留 GPU 的多个执行上下文与周期模型。GPU ISA 的公共部分是算术、访存、
分支、同步；SIMT 的线程身份、mask、divergence 和 barrier 属于额外的执行约定。
本实现使用 RISC-V 基础指令加 Vortex 自定义扩展，不是 CUDA/PTX 或 AMD ISA
二进制兼容实现。设备地址与 CPU 虚拟地址分离；当前最小软件栈不要求 GPU
MMU/PTW，但也不提供进程隔离或统一虚拟地址能力。
