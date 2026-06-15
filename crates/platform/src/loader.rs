//! Loading a real arm64 kernel: parse the `Image` header and lay out the
//! kernel, initramfs, and device tree in guest RAM without collisions.

use crate::board::{Board, RAM_BASE};

/// arm64 `Image` header magic ("ARM\x64", little-endian) at byte offset 56.
const ARM64_IMAGE_MAGIC: u32 = 0x644d_5241;
/// 2 MiB alignment for image / initrd / DTB placement.
const ALIGN_2M: u64 = 0x20_0000;
/// Fallback load offset for a header-less image (conventional 512 KiB).
const DEFAULT_TEXT_OFFSET: u64 = 0x8_0000;

fn align_up(x: u64, align: u64) -> u64 {
    (x + align - 1) & !(align - 1)
}

fn read_u64_le(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

/// The arm64 `Image` header fields the boot protocol needs (booting.rst).
#[derive(Debug, Clone, Copy)]
pub struct ImageHeader {
    /// Load offset from the 2 MiB-aligned base.
    pub text_offset: u64,
    /// Effective image size (includes BSS); 0 on very old kernels.
    pub image_size: u64,
}

/// Parse an arm64 `Image` header. Returns `None` if the magic is absent (e.g. a
/// raw, header-less test blob).
#[must_use]
pub fn parse_image_header(image: &[u8]) -> Option<ImageHeader> {
    if image.len() < 64 {
        return None;
    }
    let magic = u32::from_le_bytes(image[56..60].try_into().unwrap());
    if magic != ARM64_IMAGE_MAGIC {
        return None;
    }
    Some(ImageHeader { text_offset: read_u64_le(image, 8), image_size: read_u64_le(image, 16) })
}

/// Resulting physical placement of the loaded images.
#[derive(Debug, Clone, Copy)]
pub struct BootLayout {
    pub kernel: u64,
    pub initrd: Option<(u64, u64)>,
    pub dtb: u64,
}

impl Board {
    /// Load a real arm64 kernel `Image` (and optional initramfs), generating the
    /// device tree and laying everything out without collisions:
    ///
    /// ```text
    ///   RAM_BASE ┌─ (base, 2 MiB aligned)
    ///            │  kernel  @ base + text_offset, spanning image_size
    ///            │  initrd  @ 2 MiB-aligned, above the kernel
    ///            │  DTB     @ 2 MiB-aligned, above the initrd
    /// ```
    ///
    /// Sets up entry per the boot protocol (x0 = DTB, PC = kernel, EL1h, DAIF
    /// masked) and returns where everything landed.
    pub fn boot_image(&mut self, image: &[u8], initrd: Option<&[u8]>, bootargs: &str) -> BootLayout {
        let ram_len = self.ram_size();

        // Kernel: base (= RAM_BASE, 2 MiB aligned) + text_offset.
        let header = parse_image_header(image);
        let text_offset = header.map_or(DEFAULT_TEXT_OFFSET, |h| h.text_offset);
        let kernel_addr = RAM_BASE + text_offset;
        let span = header.map_or(0, |h| h.image_size).max(image.len() as u64);
        let kernel_end = kernel_addr + span;

        // initrd: 2 MiB-aligned, above the kernel.
        let initrd_range = initrd.map(|data| {
            let start = align_up(kernel_end, ALIGN_2M);
            (start, start + data.len() as u64)
        });

        // DTB: 2 MiB-aligned, above the initrd (or the kernel if no initrd).
        let dtb_addr = align_up(initrd_range.map_or(kernel_end, |(_, end)| end), ALIGN_2M);
        let dtb = self.dtb(ram_len, bootargs, initrd_range);

        let dtb_end = dtb_addr + dtb.len() as u64;
        assert!(
            dtb_end <= RAM_BASE + ram_len,
            "images do not fit in {ram_len:#x} of RAM (need up to {dtb_end:#x})",
        );

        // Place everything.
        let ram = self.machine.bus.ram_mut();
        ram.write(kernel_addr, image);
        if let (Some(data), Some((start, _))) = (initrd, initrd_range) {
            ram.write(start, data);
        }
        ram.write(dtb_addr, &dtb);

        self.enter(kernel_addr, dtb_addr);
        BootLayout { kernel: kernel_addr, initrd: initrd_range, dtb: dtb_addr }
    }
}
