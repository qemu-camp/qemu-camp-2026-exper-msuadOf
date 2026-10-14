#!/usr/bin/env python3
"""Fetch pinned dependencies and build the real simx/OpenCL/OS experiment."""

import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
DEPS = ROOT / "build/deps"
BUILD = ROOT / "build/vortex32"
TOOLS = ROOT / "build/vortex-tools"
OUT = ROOT / "build/gpgpu-vortex"
CONFIGS = "-DVX_CFG_NUM_CORES=4 -DVX_CFG_NUM_WARPS=4 -DVX_CFG_NUM_THREADS=32"
VORTEX = "942e381aadd9660128470dc5c0d18ad27d7e0a4b"
ARCEOS = "8d2e6f97efc4a67359b99db5e519a55b7067b524"
RCORE = "209476f27359b47cd6534ebb91afc5d01692b429"
ENV = os.environ.copy()
ENV["PATH"] = "/usr/bin:/bin:" + str(pathlib.Path.home() / ".cargo/bin")
ENV["CCACHE_DISABLE"] = "1"
ENV["CARGO_HOME"] = str(ROOT / "build/cargo")
ENV["RUSTUP_TOOLCHAIN"] = "stable"
ENV["RUSTC_BOOTSTRAP"] = "1"
ENV["GPGPU_KERNEL_DIR"] = str(OUT / "kernels")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def run(args, cwd=ROOT, env=ENV):
    print("+", " ".join(str(x) for x in args), flush=True)
    subprocess.run([str(x) for x in args], cwd=cwd, env=env, check=True)


def archive(repo, revision, dest):
    stamp = dest / ".gpgpu-source-revision"
    if stamp.exists() and stamp.read_text().strip() == revision:
        return
    if (dest / ".git").exists():
        actual = subprocess.check_output(
            ["/usr/bin/git", "-C", str(dest), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise RuntimeError(f"{dest}: expected {revision}, found {actual}")
        return
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(f"{dest} exists without a matching revision stamp")
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{repo}/tar.gz/{revision}"
    print("Fetch", url, flush=True)
    with OPENER.open(url, timeout=60) as r, tarfile.open(fileobj=r, mode="r|gz") as t:
        for m in t:
            parts = m.name.split("/", 1)
            if len(parts) == 2 and parts[1]:
                m.name = parts[1]
                t.extract(m, dest, filter="data")
    stamp.write_text(revision + "\n")


def fetch_simx():
    src = DEPS / "vortex"
    archive("vortexgpgpu/vortex", VORTEX, src)
    archive(
        "ucb-bar/berkeley-softfloat-3",
        "b51ef8f3201669b2288104c28546fc72532a1ea4",
        src / "third_party/softfloat",
    )
    archive(
        "CMU-SAFARI/ramulator2",
        "e62c84a6f0e06566ba6e182d308434b4532068a5",
        src / "third_party/ramulator",
    )
    for name, repo, rev in [
        ("yaml-cpp", "jbeder/yaml-cpp", "yaml-cpp-0.7.0"),
        ("spdlog", "gabime/spdlog", "v1.11.0"),
        ("argparse", "p-ranav/argparse", "v2.9"),
    ]:
        archive(repo, rev, src / "third_party/ramulator/ext" / name)


def toolchain():
    TOOLS.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "build/vortex-downloads"
    cache.mkdir(exist_ok=True)
    for name, remote, parts in [
        (
            "llvm-vortex",
            "llvm-vortex/ubuntu/focal/llvm-vortex.tar.bz2",
            ["aa", "ab", "ac"],
        ),
        ("pocl", "pocl/ubuntu/focal/pocl.tar.bz2", [""]),
        ("libcrt32", "libcrt32/libcrt32.tar.bz2", [""]),
        ("libc32", "libc32/libc32.tar.bz2", [""]),
    ]:
        if (TOOLS / name).exists():
            continue
        tar = cache / (name + ".tar.bz2")
        with tar.open("wb") as output:
            for part in parts:
                url = (
                    "https://raw.githubusercontent.com/vortexgpgpu/vortex-toolchain-prebuilt/v3.0.1/"
                    + remote
                )
                if part:
                    url += ".part" + part
                with OPENER.open(url, timeout=120) as r:
                    shutil.copyfileobj(r, output)
        digest = hashlib.sha256(tar.read_bytes()).hexdigest()
        (cache / (name + ".sha256")).write_text(digest + "\n")
        with tarfile.open(tar) as t:
            t.extractall(TOOLS, filter="data")
    icd = TOOLS / "pocl/etc/OpenCL/vendors/pocl.icd"
    icd.parent.mkdir(parents=True, exist_ok=True)
    icd.write_text(str(TOOLS / "pocl/lib/libpocl.so.2.14.0") + "\n")


def simx():
    BUILD.mkdir(parents=True, exist_ok=True)
    run(
        [
            DEPS / "vortex/configure",
            "--xlen=32",
            f"--tooldir={TOOLS}",
            "--osversion=ubuntu/focal",
        ],
        BUILD,
    )
    run(
        [
            "make",
            "-C",
            DEPS / "vortex/third_party/softfloat/build/Linux-x86_64-GCC",
            f"-j{A.jobs}",
        ]
    )
    ram = ROOT / "build/vortex-ramulator"
    run(
        [
            "cmake",
            "-S",
            DEPS / "vortex/third_party/ramulator",
            "-B",
            ram,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        ]
    )
    run(["cmake", "--build", ram, f"-j{A.jobs}"])
    run(["make", "-C", DEPS / "vortex/third_party", "ramulator"])
    run(
        [
            "make",
            "-C",
            BUILD / "sim/simx",
            f"-j{A.jobs}",
            "CXX=g++-11",
            f"CONFIGS={CONFIGS}",
            BUILD / "sim/simx/libsimx.so",
        ]
    )
    run(["make", "-C", HERE, f"-j{A.jobs}"])


def opencl():
    archive("KhronosGroup/OpenCL-Headers", "v2024.10.24", DEPS / "opencl-headers")
    run(
        [
            "make",
            "-C",
            BUILD / "sw/kernel",
            f"-j{A.jobs}",
            f"RISCV_TOOLCHAIN_PATH={A.riscv_gnu}",
            "RISCV_PREFIX=riscv64-unknown-elf",
            f"CONFIGS={CONFIGS}",
        ]
    )
    run(["make", "-C", BUILD / "sw/runtime/stub", f"-j{A.jobs}", f"CONFIGS={CONFIGS}"])
    run(["make", "-C", HERE, f"-j{A.jobs}"])
    app = BUILD / "tests/opencl/sgemm"
    run(
        [
            "g++-11",
            "-O2",
            "-std=c++17",
            "-DCL_TARGET_OPENCL_VERSION=120",
            f"-I{DEPS}/opencl-headers",
            DEPS / "vortex/tests/opencl/sgemm/main.cc",
            "-Wl,-l:libOpenCL.so.1",
            "-o",
            app / "sgemm",
        ]
    )
    for name in ["common.h", "kernel.cl"]:
        shutil.copy2(DEPS / "vortex/tests/opencl/sgemm" / name, app / name)


def os_clients():
    archive("arceos-org/arceos", ARCEOS, DEPS / "arceos")
    archive("rcore-os/rCore-Tutorial-v3", RCORE, DEPS / "rcore")
    for name, repo, rev in [
        ("rust-fatfs", "rafalh/rust-fatfs", "4eccb50d011146fbed20e133d33b22f3c27292e7"),
        ("smoltcp", "rcore-os/smoltcp", "21a2f82b062ec88442eb624d7fcec1feeb886496"),
        (
            "lose-net-stack",
            "yfblock/lose-net-stack",
            "db4238008f0bb97f0b82550ef949c9a32ead59d2",
        ),
        ("virtio-drivers", "rcore-os/virtio-drivers", "4ee80e5"),
    ]:
        archive(repo, rev, DEPS / name)
    run(
        [
            "python3",
            HERE / "prepare-os.py",
            "--arceos",
            DEPS / "arceos",
            "--rcore",
            DEPS / "rcore",
        ]
    )
    bin_dir = ROOT / "build/arceos-tools"
    env = ENV.copy()
    env["PATH"] = f"{bin_dir}/bin:{ENV['CARGO_HOME']}/bin:" + ENV["PATH"]
    if not (bin_dir / "bin/axconfig-gen").exists():
        run(
            [
                "cargo",
                "install",
                "axconfig-gen",
                "--version",
                "0.2.1",
                "--locked",
                "--root",
                bin_dir,
            ],
            env=env,
        )
    run(
        [
            "make",
            "-C",
            DEPS / "arceos",
            "ARCH=riscv64",
            "A=examples/helloworld",
            "SMP=1",
            "OBJCOPY=riscv64-unknown-elf-objcopy",
            f"OUT_DIR={OUT}/arceos",
            f"TARGET_DIR={OUT}/arceos-target",
            f"OUT_CONFIG={OUT}/arceos-config.toml",
        ],
        env=env,
    )
    run(["cargo", "build", "--release"], DEPS / "rcore/os", env)


parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["fetch", "tools", "simx", "opencl", "os", "all"])
parser.add_argument("--jobs", type=int, default=16)
parser.add_argument(
    "--riscv-gnu",
    type=pathlib.Path,
    default=pathlib.Path("/home/baiyifan/riscv/riscv64-multilib"),
)
A = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
DEPS.mkdir(parents=True, exist_ok=True)
if A.phase in ["fetch", "all"]:
    fetch_simx()
if A.phase in ["tools", "all"]:
    toolchain()
if A.phase in ["simx", "all"]:
    simx()
if A.phase in ["opencl", "all"]:
    opencl()
if A.phase in ["os", "all"]:
    os_clients()
