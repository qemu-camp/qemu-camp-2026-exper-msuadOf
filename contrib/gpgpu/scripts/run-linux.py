#!/usr/bin/env python3
"""Boot a Linux guest and require successful real GPGPU driver/runtime demos."""
import argparse
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qemu", type=pathlib.Path, required=True)
    parser.add_argument("--kernel", type=pathlib.Path, required=True)
    parser.add_argument("--initrd", type=pathlib.Path, required=True)
    parser.add_argument("--bios", type=pathlib.Path)
    parser.add_argument("--device", default="gpgpu")
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    command = [str(args.qemu), "-M", "virt", "-m", "512M", "-smp", "1",
               "-nographic", "-monitor", "none", "-nic", "none",
               "-kernel", str(args.kernel), "-initrd", str(args.initrd),
               "-append", "console=ttyS0 rdinit=/init panic=-1",
               "-device", args.device]
    if args.bios:
        command += ["-bios", str(args.bios)]
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=args.timeout)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_bytes(result.stdout)
    text = result.stdout.decode(errors="replace")
    for marker in ["vector add: PASS", "ReLU: PASS",
                   "matrix multiplication: PASS", "BAR2 mmap: PASS",
                   "GPGPU SDK: ALL DEMOS PASSED", "GPGPU DEMO EXIT: 0"]:
        if marker not in text:
            raise SystemExit(f"Missing {marker!r}; inspect {args.log}")
        print(marker)
    if result.returncode:
        raise SystemExit(f"QEMU exited with {result.returncode}")


if __name__ == "__main__":
    main()
