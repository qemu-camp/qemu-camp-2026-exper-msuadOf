#!/usr/bin/env python3
"""Install small GPGPU demos into pinned, disposable OS checkouts."""

import argparse
import pathlib
import shutil
import re

parser = argparse.ArgumentParser()
parser.add_argument("--arceos", type=pathlib.Path, required=True)
parser.add_argument("--rcore", type=pathlib.Path, required=True)
a = parser.parse_args()
here = pathlib.Path(__file__).resolve().parent
app = a.arceos / "examples/helloworld"
for dest in [app / "src/gpgpu.rs", a.rcore / "os/src/gpgpu.rs"]:
    shutil.copy2(here / "os_driver.rs", dest)
(app / "src/main.rs").write_text("""#![no_std]
#![no_main]
use axstd::println;
mod gpgpu;
#[unsafe(no_mangle)]
fn main() {
    gpgpu::exercise(|p| axhal::mem::phys_to_virt(p.into()).as_usize(),
                    |v| axhal::mem::virt_to_phys(v.into()).as_usize());
    println!("ArceOS GPGPU: PCI DMA IRQ vecadd barrier divergence PASS");
}
""")
p = app / "Cargo.toml"
text = p.read_text()
if "axhal =" not in text:
    p.write_text(text + "\naxhal = { workspace = true }\n")
text = p.read_text().replace(
    "axstd = { workspace = true, optional = true }",
    'axstd = { workspace = true, optional = true, features = ["alloc", "paging"] }',
)
p.write_text(text)
p = a.rcore / "os/src/main.rs"
text = p.read_text()
if "mod gpgpu;" not in text:
    text = text.replace(
        "    logging::init();",
        """    logging::init();
    gpgpu::exercise(|p| p, |v| v);
    println!("rCore GPGPU: PCI DMA IRQ vecadd barrier divergence PASS");
    unsafe { core::ptr::write_volatile(0x100000 as *mut u32, 0x5555); }
    loop { core::hint::spin_loop(); }
""",
        1,
    )
    # Module items must follow the crate-level attributes.
    text = text.replace("extern crate alloc;", "extern crate alloc;\nmod gpgpu;", 1)
    p.write_text(text)
p = a.rcore / "os/src/boards/qemu.rs"
text = p.read_text()
if "0x30000000" not in text:
    text = text.replace(
        "pub const MMIO: &[(usize, usize)] = &[",
        """pub const MMIO: &[(usize, usize)] = &[
    (0x30000000, 0x200000),
    (0x40000000, 0x200000),""",
        1,
    )
    p.write_text(text)
shutil.copy2(a.rcore / "os/src/linker-qemu.ld", a.rcore / "os/src/linker.ld")
# The archive downloads preserve upstream commits while avoiding Git transport.
deps = a.arceos.parent.resolve()
for manifest, name in [
    (a.arceos / "modules/axfs/Cargo.toml", "rust-fatfs"),
    (a.arceos / "modules/axnet/Cargo.toml", "smoltcp"),
]:
    text = manifest.read_text()
    text = re.sub(
        r'git = "https://github.com/[^"\n]+"\nrev = "[^"\n]+"',
        f'path = "{deps / name}"',
        text,
    )
    manifest.write_text(text)
manifest = a.rcore / "os/Cargo.toml"
text = manifest.read_text()
for name in ["virtio-drivers", "lose-net-stack"]:
    text = re.sub(
        rf'({name} = \{{ )git = "[^"\n]+", rev = "[^"\n]+"',
        rf'\1path = "{deps / name}"',
        text,
    )
manifest.write_text(text)
