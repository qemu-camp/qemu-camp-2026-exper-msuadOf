#!/usr/bin/env python3
"""Exercise the real simx backend through PCI MMIO, including MSI-X."""

import argparse
import pathlib
import socket
import struct
import subprocess
import tempfile
import time

p = argparse.ArgumentParser()
p.add_argument(
    "--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[3]
)
a = p.parse_args()
root = a.root.resolve()
out = root / "build/gpgpu-vortex"
with tempfile.TemporaryDirectory(prefix="gpgpu-simx-", dir="/tmp") as tmp:
    path = pathlib.Path(tmp) / "qtest.sock"
    with (out / "simx-qemu.log").open("wb") as log:
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
                f"unix:{path},server=on,wait=off",
                "-qtest-log",
                "/dev/null",
            ],
            stdout=log,
            stderr=log,
            cwd=out,
        )
        try:
            for _ in range(100):
                if path.exists():
                    break
                if q.poll() is not None:
                    raise RuntimeError("QEMU startup failed")
                time.sleep(0.05)
            s = socket.socket(socket.AF_UNIX)
            s.settimeout(30)
            s.connect(str(path))
            stream = s.makefile("rwb", buffering=0)

            def cmd(text):
                stream.write((text + "\n").encode())
                line = stream.readline().decode().strip()
                assert line.startswith("OK"), line
                return line[3:]

            def rd(addr):
                return int(cmd(f"readl {addr:#x}"), 0)

            def wr(addr, value):
                cmd(f"writel {addr:#x} {value:#x}")

            def memwrite(addr, data):
                cmd(f"write {addr:#x} {len(data):#x} 0x{data.hex()}")

            def memread(addr, n):
                return bytes.fromhex(cmd(f"read {addr:#x} {n:#x}")[2:])

            cfg = next(
                0x30000000 + (slot << 15)
                for slot in range(32)
                if rd(0x30000000 + (slot << 15)) == 0x13371234
            )
            for r, v in [
                (0x10, 0x40000000),
                (0x14, 0),
                (0x18, 0x44000000),
                (0x1C, 0),
                (0x20, 0x40100000),
                (4, 6),
            ]:
                wr(cfg + r, v)
            ctrl = 0x40000000
            vram = 0x44000000
            door = 0x40100000
            wr(ctrl + 0x100, 1)
            wr(ctrl + 0x200, 7)
            cap = rd(cfg + 0x34) & 255
            while rd(cfg + cap) & 255 != 0x11:
                cap = (rd(cfg + cap) >> 8) & 255
            cmd(f"writew {cfg + cap + 2:#x} 0x8003")
            for vector in range(3):
                addr = ctrl + 0xFE000 + vector * 16
                for off, val in [
                    (0, 0x81000000 + vector * 4),
                    (4, 0),
                    (8, 0xDEC00000 + vector),
                    (12, 0),
                ]:
                    wr(addr + off, val)

            def launch(name, args, grid, block, shared=0):
                wr(ctrl + 0x208, 7)
                memwrite(0x81000000, bytes(12))
                memwrite(vram + 0x10000, (out / f"kernels/{name}.bin").read_bytes())
                memwrite(vram + 0x110000, args)
                for r, v in [
                    (0x300, 0x10000),
                    (0x304, 0),
                    (0x308, 0x110000),
                    (0x30C, 0),
                    (0x328, shared),
                ]:
                    wr(ctrl + r, v)
                for i in range(3):
                    wr(ctrl + 0x310 + i * 4, grid[i])
                    wr(ctrl + 0x31C + i * 4, block[i])
                wr(ctrl + 0x330, 1)
                for _ in range(100000):
                    if rd(ctrl + 0x104) & 2 == 0:
                        break
                    cmd("clock_step 10000")
                else:
                    raise AssertionError("simx timeout")
                assert rd(ctrl + 0x108) == 0, (name, rd(ctrl + 0x108))
                assert rd(ctrl + 0x204) & 1
                assert rd(0x81000000) == 0xDEC00000, (
                    "missing real simx MSI-X completion"
                )
                assert rd(0x81000008) == 0, "unexpected error interrupt"
                return rd(door + 0x2000), rd(door + 0x2004), rd(door + 0x2008)

            args = struct.pack("<4I", 0x100000, 0x101000, 0x102000, 64)
            for kernel, shared in [("sync", 0), ("local", 256)]:
                cycles, requests, divergence = launch(
                    kernel, args, [1, 1, 1], [64, 1, 1], shared
                )
                result = struct.unpack("<64I", memread(vram + 0x102000, 256))
                expect = [
                    (((i + 32) % 64 + 1) * 3 if i & 1 else ((i + 32) % 64 + 1) + 7)
                    for i in range(64)
                ]
                assert list(result) == expect, (kernel, result)
                assert cycles > 0 and requests > 0 and divergence > 0
                print(
                    f"{kernel}: 64 lanes / two warps / barrier / divergence PASS; cycles={cycles}, memory requests={requests}, divergence={divergence}"
                )
            n = 8
            aa = [float(i % 11 - 5) for i in range(64)]
            bb = [float(i % 7 - 3) for i in range(64)]
            memwrite(vram + 0x100000, struct.pack("<64f", *aa))
            memwrite(vram + 0x101000, struct.pack("<64f", *bb))
            args = struct.pack("<I4xQQQ", n, 0x100000, 0x101000, 0x102000)
            launch("sgemm", args, [2, 2, 1], [4, 4, 1])
            cc = struct.unpack("<64f", memread(vram + 0x102000, 256))
            assert list(cc) == [
                sum(aa[r * n + k] * bb[k * n + c] for k in range(n))
                for r in range(n)
                for c in range(n)
            ]
            print("upstream native sgemm: PASS (64 results)")
            aa = [float(i - 300) for i in range(576)]
            bb = [float(2 * i) for i in range(576)]
            memwrite(vram + 0x100000, struct.pack("<576f", *aa))
            memwrite(vram + 0x101000, struct.pack("<576f", *bb))
            memwrite(vram + 0x102000, struct.pack("<576f", *([-999.0] * 576)))
            args = struct.pack("<4I", 0x100000, 0x101000, 0x102000, 550)
            launch("vecadd", args, [9, 1, 1], [64, 1, 1])
            cc = struct.unpack("<576f", memread(vram + 0x102000, 576 * 4))
            assert list(cc) == [
                aa[i] + bb[i] if i < 550 else -999.0 for i in range(576)
            ]
            print(
                "CTA reuse, nine blocks, partial final warp: PASS (550 results / 26 untouched lanes)"
            )
            # Invalid device addresses must produce ERROR, not terminate QEMU.
            memwrite(
                vram + 0x110000, struct.pack("<4I", 0x70000000, 0x101000, 0x102000, 64)
            )
            memwrite(vram + 0x10000, (out / "kernels/vecadd.bin").read_bytes())
            wr(ctrl + 0x208, 7)
            wr(ctrl + 0x330, 1)
            for _ in range(10000):
                if rd(ctrl + 0x104) & 2 == 0:
                    break
                cmd("clock_step 10000")
            assert rd(ctrl + 0x108) & 4
            assert rd(0x81000008) == 0xDEC00002, "missing simx error MSI-X"
            wr(ctrl + 0x100, 2)
            wr(ctrl + 0x200, 7)
            wr(ctrl + 0x100, 1)
            args = struct.pack("<4I", 0x100000, 0x101000, 0x102000, 64)
            launch("sync", args, [1, 1, 1], [64, 1, 1])
            print("MSI-X completion/error and fault/reset recovery: PASS")
            stream.close()
            s.close()
        finally:
            q.terminate()
            try:
                q.wait(timeout=10)
            except subprocess.TimeoutExpired:
                q.kill()
                q.wait()
