#!/usr/bin/env python3
"""Run the upstream OpenCL sgemm against QEMU's embedded simx."""

import argparse
import os
import pathlib
import subprocess
import tempfile
import time

p = argparse.ArgumentParser()
p.add_argument(
    "--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[3]
)
p.add_argument("--size", type=int, default=32)
p.add_argument("--timeout", type=int, default=180)
p.add_argument(
    "--riscv-gnu",
    type=pathlib.Path,
    default=pathlib.Path("/home/baiyifan/riscv/riscv64-multilib"),
)
a = p.parse_args()
root = a.root.resolve()
b = root / "build/vortex32"
src = root / "build/deps/vortex"
tools = root / "build/vortex-tools"
out = root / "build/gpgpu-vortex"
cfg = "-DVX_CFG_NUM_CORES=4 -DVX_CFG_NUM_WARPS=4 -DVX_CFG_NUM_THREADS=32"
with tempfile.TemporaryDirectory(prefix="gpgpu-pocl-", dir="/tmp") as tmp:
    sock = pathlib.Path(tmp) / "qtest.sock"
    log = (out / "opencl-qemu.log").open("wb")
    q = subprocess.Popen(
        [
            str(root / "build/qemu-system-riscv64"),
            "-M",
            "virt,accel=qtest",
            "-m",
            "256M",
            "-display",
            "none",
            "-monitor",
            "none",
            "-serial",
            "none",
            "-nic",
            "none",
            "-device",
            f"gpgpu,backend=simx,simx_library={out}/libgpgpu-vortex.so",
            "-qtest",
            f"unix:{sock},server=on,wait=off",
            "-qtest-log",
            "/dev/null",
        ],
        stdout=log,
        stderr=log,
        cwd=out,
    )
    try:
        for _ in range(100):
            if sock.exists():
                break
            if q.poll() is not None:
                raise RuntimeError("QEMU startup failed")
            time.sleep(0.05)
        env = os.environ.copy()
        flags = (
            f"--target=riscv32-unknown-elf -march=rv32imf -mabi=ilp32f -O3 -mcmodel=medany "
            f"--sysroot={tools}/libc32 --gcc-toolchain={a.riscv_gnu.resolve()} "
            f"-fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections "
            f"-I{b}/sw -I{b}/hw -I{src}/sw/kernel/include -DVX_CFG_XLEN=32 -DVX_CFG_XLEN_32 "
            f"{cfg} -D__VORTEX__ -Xclang -target-feature -Xclang +xvortex "
            "-Xclang -target-feature -Xclang +zicond -mllvm -disable-loop-idiom-all"
        )
        ld = (
            f"--ld-path=/usr/bin/ld.lld -Wl,-z,norelro -Wl,-Bstatic,--gc-sections,"
            f"-T{src}/sw/kernel/scripts/link32.ld,--defsym=STARTUP_ADDR=0x10000 "
            f"{b}/sw/kernel/libvortex2.a {out}/crt.o -L{tools}/libc32/lib -lm -lc "
            f"{tools}/libcrt32/lib/baremetal/libclang_rt.builtins-riscv32.a"
        )
        env.update(
            {
                "GPGPU_QTEST_SOCKET": str(sock),
                "VORTEX_DRIVER": "qemu",
                "POCL_DEVICES": "vortex",
                "LD_LIBRARY_PATH": f"{tools}/pocl/lib:{b}/sw/runtime:{tools}/llvm-vortex/lib",
                "OCL_ICD_VENDORS": str(tools / "pocl/etc/OpenCL/vendors"),
                "LLVM_PREFIX": str(tools / "llvm-vortex"),
                "POCL_VORTEX_XLEN": "32",
                "POCL_VORTEX_CFLAGS": flags,
                "POCL_VORTEX_LDFLAGS": ld,
                "POCL_VORTEX_BINTOOL": f"OBJCOPY={tools}/llvm-vortex/bin/llvm-objcopy {src}/sw/kernel/scripts/vxbin.py",
                "POCL_CACHE_DIR": str(out / "pocl-cache"),
            }
        )
        run = subprocess.run(
            [str(b / "tests/opencl/sgemm/sgemm"), f"-n{a.size}"],
            cwd=b / "tests/opencl/sgemm",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=a.timeout,
        )
        (out / "opencl-sgemm.log").write_bytes(run.stdout)
        print(run.stdout.decode(errors="replace"))
        if run.returncode:
            raise SystemExit(run.returncode)
        if b"PASSED" not in run.stdout:
            raise SystemExit("sgemm did not report PASSED")
    finally:
        q.terminate()
        try:
            q.wait(timeout=10)
        except subprocess.TimeoutExpired:
            q.kill()
            q.wait()
        log.close()
