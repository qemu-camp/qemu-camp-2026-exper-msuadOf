#!/usr/bin/env python3
"""Boot ArceOS/rCore and require the real PCI/DMA/PLIC GPGPU demo."""

import argparse
import pathlib
import subprocess

p = argparse.ArgumentParser()
p.add_argument("os", choices=["arceos", "rcore"])
p.add_argument(
    "--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[3]
)
p.add_argument("--timeout", type=int, default=90)
a = p.parse_args()
root = a.root.resolve()
out = root / "build/gpgpu-vortex"
kernel = (
    out / "arceos/helloworld_riscv64-qemu-virt.bin"
    if a.os == "arceos"
    else root / "build/deps/rcore/os/target/riscv64gc-unknown-none-elf/release/os"
)
cmd = [
    str(root / "build/qemu-system-riscv64"),
    "-M",
    "virt",
    "-m",
    "128M",
    "-smp",
    "1",
    "-nographic",
    "-monitor",
    "none",
    "-nic",
    "none",
    "-bios",
    str(root / "pc-bios/opensbi-riscv64-generic-fw_dynamic.bin"),
    "-kernel",
    str(kernel),
    "-device",
    f"gpgpu,backend=simx,simx_library={out}/libgpgpu-vortex.so",
]
try:
    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=a.timeout,
        cwd=out,
    )
    data = r.stdout
except subprocess.TimeoutExpired as e:
    data = e.stdout or b""
    (out / f"{a.os}-demo.log").write_bytes(data)
    raise SystemExit(f"{a.os} timed out; inspect {out}/{a.os}-demo.log")
(out / f"{a.os}-demo.log").write_bytes(data)
marker = (
    "ArceOS" if a.os == "arceos" else "rCore"
) + " GPGPU: PCI DMA IRQ vecadd barrier divergence PASS"
if r.returncode or marker.encode() not in data:
    print(data.decode(errors="replace"))
    raise SystemExit("OS GPGPU validation failed")
print(marker)
