// SPDX-License-Identifier: MIT
use core::ptr::{read_volatile, write_volatile};

pub struct Gpgpu {
    regs: usize,
    phys_to_virt: fn(usize) -> usize,
    virt_to_phys: fn(usize) -> usize,
}
impl Gpgpu {
    fn load(&self, p: usize) -> u32 {
        unsafe { read_volatile(p as *const u32) }
    }
    fn store(&self, p: usize, v: u32) {
        unsafe { write_volatile(p as *mut u32, v) }
    }
    fn reg(&self, r: usize) -> u32 {
        self.load(self.regs + r)
    }
    fn set(&self, r: usize, v: u32) {
        self.store(self.regs + r, v);
    }
    pub fn probe(p2v: fn(usize) -> usize, v2p: fn(usize) -> usize) -> Self {
        let dev = Self {
            regs: p2v(0x40000000),
            phys_to_virt: p2v,
            virt_to_phys: v2p,
        };
        let ecam = p2v(0x30000000);
        let cfg = (0..32)
            .map(|s| ecam + (s << 15))
            .find(|a| dev.load(*a) == 0x13371234)
            .expect("GPGPU PCI device missing");
        for (r, v) in [
            (0x10, 0x40000000),
            (0x14, 0),
            (0x18, 0x44000000),
            (0x1c, 0),
            (0x20, 0x40100000),
            (4, 6),
        ] {
            dev.store(cfg + r, v);
        }
        assert_eq!(dev.reg(0), 0x47505055);
        assert_eq!(dev.reg(0xc), 64 * 1024 * 1024);
        dev.set(0x100, 2);
        dev.set(0x200, 7);
        dev.set(0x100, 1);
        let plic = p2v(0xc000000);
        for irq in 32..36 {
            dev.store(plic + irq * 4, 1);
        }
        let enable = plic + 0x2084;
        dev.store(enable, dev.load(enable) | 15);
        dev.store(plic + 0x201000, 0);
        dev
    }
    fn fence() {
        unsafe {
            core::arch::asm!("fence iorw, iorw", options(nostack));
        }
    }
    fn wait_irq(&self, event: u32) {
        for _ in 0..20_000_000 {
            let irq = self.reg(0x204);
            if irq & (event | 4) != 0 {
                assert_eq!(self.reg(0x108), 0, "GPU error");
                let claim = (self.phys_to_virt)(0xc201004);
                let n = self.load(claim);
                assert!(
                    (32..36).contains(&n),
                    "GPGPU INTx not delivered through PLIC"
                );
                self.set(0x208, irq);
                self.store(claim, n);
                return;
            }
            core::hint::spin_loop();
        }
        panic!("GPGPU timeout");
    }
    pub fn copy(&self, device: u32, host: &mut [u8], from_device: bool) {
        let addr = (self.virt_to_phys)(host.as_ptr() as usize) as u64;
        let (src, dst) = if from_device {
            (device as u64, addr)
        } else {
            (addr, device as u64)
        };
        self.set(0x400, src as u32);
        self.set(0x404, (src >> 32) as u32);
        self.set(0x408, dst as u32);
        self.set(0x40c, (dst >> 32) as u32);
        self.set(0x410, host.len() as u32);
        Self::fence();
        self.set(0x414, 5 | if from_device { 2 } else { 0 });
        self.wait_irq(2);
        Self::fence();
        assert_eq!(self.reg(0x418), 2);
    }
    pub fn launch(&self, image: &[u8], args: &mut [u8], grid: [u32; 3], block: [u32; 3]) {
        // Copies use RAM-backed buffers, never the GPU BAR as a DMA source.
        let mut chunk = [0u8; 4096];
        for (i, part) in image.chunks(4096).enumerate() {
            chunk[..part.len()].copy_from_slice(part);
            self.copy(0x10000 + (i * 4096) as u32, &mut chunk[..part.len()], false);
        }
        self.copy(0x110000, args, false);
        self.set(0x300, 0x10000);
        self.set(0x304, 0);
        self.set(0x308, 0x110000);
        self.set(0x30c, 0);
        for i in 0..3 {
            self.set(0x310 + i * 4, grid[i]);
            self.set(0x31c + i * 4, block[i]);
        }
        self.set(0x330, 1);
        self.wait_irq(1);
    }
}

pub fn exercise(p2v: fn(usize) -> usize, v2p: fn(usize) -> usize) {
    let gpu = Gpgpu::probe(p2v, v2p);
    let mut a = [0u8; 256];
    let mut b = [0u8; 256];
    let mut c = [0u8; 256];
    for i in 0..64 {
        a[i * 4..i * 4 + 4].copy_from_slice(&((i as i32 - 32) as f32).to_le_bytes());
        b[i * 4..i * 4 + 4].copy_from_slice(&((2 * i) as f32).to_le_bytes());
    }
    gpu.copy(0x100000, &mut a, false);
    gpu.copy(0x101000, &mut b, false);
    let mut args = [0u8; 16];
    for (i, x) in [0x100000u32, 0x101000, 0x102000, 64].iter().enumerate() {
        args[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
    gpu.launch(
        include_bytes!(concat!(env!("GPGPU_KERNEL_DIR"), "/vecadd.bin")),
        &mut args,
        [2, 1, 1],
        [32, 1, 1],
    );
    gpu.copy(0x102000, &mut c, true);
    for i in 0..64 {
        assert_eq!(
            f32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap()),
            (3 * i as i32 - 32) as f32
        );
    }
    gpu.launch(
        include_bytes!(concat!(env!("GPGPU_KERNEL_DIR"), "/sync.bin")),
        &mut args,
        [1, 1, 1],
        [64, 1, 1],
    );
    gpu.copy(0x102000, &mut c, true);
    for i in 0..64 {
        let other = ((i + 32) % 64 + 1) as u32;
        let expected = if i & 1 != 0 { other * 3 } else { other + 7 };
        assert_eq!(
            u32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap()),
            expected
        );
    }
}
