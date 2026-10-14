#!/usr/bin/env python3
"""Package the GPGPU driver, SDK demos and static BusyBox for a Linux guest."""
import argparse
import pathlib
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", type=pathlib.Path, required=True)
    parser.add_argument("--busybox", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        root = pathlib.Path(temp)
        for directory in ["bin", "dev", "proc", "sys", "kernels", "lib"]:
            (root / directory).mkdir()
        shutil.copy2(args.busybox, root / "bin/busybox")
        for applet in ["sh", "mount", "insmod", "poweroff"]:
            (root / "bin" / applet).symlink_to("busybox")
        shutil.copy2(args.sdk / "gpgpu-demo", root / "bin/gpgpu-demo")
        shutil.copy2(args.sdk / "driver/gpgpu_pci.ko", root / "lib/gpgpu_pci.ko")
        for image in (args.sdk / "kernels").glob("*.bin"):
            shutil.copy2(image, root / "kernels" / image.name)
        init = root / "init"
        init.write_text("""#!/bin/sh
mount -t devtmpfs devtmpfs /dev
mount -t proc proc /proc
mount -t sysfs sysfs /sys
insmod /lib/gpgpu_pci.ko || { echo 'GPGPU DRIVER FAILED'; poweroff -f; }
/bin/gpgpu-demo /kernels
status=$?
echo "GPGPU DEMO EXIT: $status"
poweroff -f
""")
        init.chmod(0o755)
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
        with args.output.open("wb") as output:
            subprocess.run(["cpio", "--null", "-o", "--format=newc"],
                           input=("\0".join(files) + "\0").encode(),
                           cwd=root, stdout=output, check=True)


if __name__ == "__main__":
    main()
