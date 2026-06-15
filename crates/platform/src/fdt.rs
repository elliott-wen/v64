//! Flattened Device Tree (DTB) generation, in Rust — no `dtc` dependency, so it
//! works natively and in wasm.
//!
//! The kernel discovers all hardware from this blob: RAM size, the CPU and its
//! `enable-method`, PSCI, the GIC, the architected timer, and the PL011 console.
//! [`FdtBuilder`] emits the binary format (DTB spec v17, big-endian); [`virt_dtb`]
//! assembles the node set matching the board this crate emulates.

use std::collections::HashMap;

// Structure-block tokens.
const FDT_BEGIN_NODE: u32 = 0x1;
const FDT_END_NODE: u32 = 0x2;
const FDT_PROP: u32 = 0x3;
const FDT_END: u32 = 0x9;

const FDT_MAGIC: u32 = 0xd00d_feed;
const FDT_VERSION: u32 = 17;
const FDT_LAST_COMP_VERSION: u32 = 16;

/// Builds a flattened device tree incrementally. Call [`Self::begin_node`] /
/// [`Self::end_node`] to nest, the `prop_*` helpers for properties, and
/// [`Self::finish`] to produce the blob.
pub struct FdtBuilder {
    structure: Vec<u8>,
    strings: Vec<u8>,
    /// Dedupes property-name -> offset into the strings block.
    string_offsets: HashMap<String, u32>,
}

impl FdtBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self { structure: Vec::new(), strings: Vec::new(), string_offsets: HashMap::new() }
    }

    fn pad_to_4(v: &mut Vec<u8>) {
        while !v.len().is_multiple_of(4) {
            v.push(0);
        }
    }

    /// Intern a property name, returning its offset in the strings block.
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(off) = self.string_offsets.get(name) {
            return *off;
        }
        let off = self.strings.len() as u32;
        self.strings.extend_from_slice(name.as_bytes());
        self.strings.push(0);
        self.string_offsets.insert(name.to_string(), off);
        off
    }

    /// Open a node. `name` is empty for the root.
    pub fn begin_node(&mut self, name: &str) {
        self.structure.extend_from_slice(&FDT_BEGIN_NODE.to_be_bytes());
        self.structure.extend_from_slice(name.as_bytes());
        self.structure.push(0);
        Self::pad_to_4(&mut self.structure);
    }

    /// Close the most recently opened node.
    pub fn end_node(&mut self) {
        self.structure.extend_from_slice(&FDT_END_NODE.to_be_bytes());
    }

    /// Raw property: token, length, name offset, value, padding.
    pub fn prop(&mut self, name: &str, value: &[u8]) {
        let nameoff = self.intern(name);
        self.structure.extend_from_slice(&FDT_PROP.to_be_bytes());
        self.structure.extend_from_slice(&(value.len() as u32).to_be_bytes());
        self.structure.extend_from_slice(&nameoff.to_be_bytes());
        self.structure.extend_from_slice(value);
        Self::pad_to_4(&mut self.structure);
    }

    /// A property with no value (e.g. `interrupt-controller`).
    pub fn prop_empty(&mut self, name: &str) {
        self.prop(name, &[]);
    }

    pub fn prop_u32(&mut self, name: &str, v: u32) {
        self.prop(name, &v.to_be_bytes());
    }

    pub fn prop_u64(&mut self, name: &str, v: u64) {
        self.prop(name, &v.to_be_bytes());
    }

    /// A `<u32 u32 ...>` cell array.
    pub fn prop_cells(&mut self, name: &str, cells: &[u32]) {
        let mut bytes = Vec::with_capacity(cells.len() * 4);
        for c in cells {
            bytes.extend_from_slice(&c.to_be_bytes());
        }
        self.prop(name, &bytes);
    }

    /// A null-terminated string property.
    pub fn prop_str(&mut self, name: &str, s: &str) {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        self.prop(name, &bytes);
    }

    /// A `<stringlist>` (concatenated null-terminated strings).
    pub fn prop_strlist(&mut self, name: &str, items: &[&str]) {
        let mut bytes = Vec::new();
        for s in items {
            bytes.extend_from_slice(s.as_bytes());
            bytes.push(0);
        }
        self.prop(name, &bytes);
    }

    /// Finalize: append `FDT_END`, lay out header + reservation map + structure
    /// + strings, and patch the offsets/sizes. Returns the blob.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.structure.extend_from_slice(&FDT_END.to_be_bytes());

        const HEADER_LEN: u32 = 40;
        // One empty (address=0, size=0) reservation entry: 16 bytes, 8-aligned.
        let rsvmap = [0u8; 16];

        let off_mem_rsvmap = HEADER_LEN;
        let off_dt_struct = off_mem_rsvmap + rsvmap.len() as u32;
        let size_dt_struct = self.structure.len() as u32;
        let off_dt_strings = off_dt_struct + size_dt_struct;
        let size_dt_strings = self.strings.len() as u32;
        let totalsize = off_dt_strings + size_dt_strings;

        let mut blob = Vec::with_capacity(totalsize as usize);
        for word in [
            FDT_MAGIC,
            totalsize,
            off_dt_struct,
            off_dt_strings,
            off_mem_rsvmap,
            FDT_VERSION,
            FDT_LAST_COMP_VERSION,
            0, // boot_cpuid_phys
            size_dt_strings,
            size_dt_struct,
        ] {
            blob.extend_from_slice(&word.to_be_bytes());
        }
        blob.extend_from_slice(&rsvmap);
        blob.extend_from_slice(&self.structure);
        blob.extend_from_slice(&self.strings);
        blob
    }
}

impl Default for FdtBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// GIC interrupt-specifier types (`#interrupt-cells = <3>`).
const GIC_SPI: u32 = 0;
const GIC_PPI: u32 = 1;
const IRQ_LEVEL_HIGH: u32 = 4;
const IRQ_LEVEL_LOW: u32 = 8;

// Phandles referenced within the tree.
const PHANDLE_GIC: u32 = 1;
const PHANDLE_CLK: u32 = 2;

/// Board parameters the device tree must describe.
pub struct DtbConfig<'a> {
    pub mem_base: u64,
    pub mem_size: u64,
    pub gicd_base: u64,
    pub gicc_base: u64,
    pub uart_base: u64,
    /// UART interrupt as an absolute GIC ID (e.g. 33 == SPI 1).
    pub uart_irq: u32,
    pub bootargs: &'a str,
    /// `(start, end)` physical addresses of an initramfs, if any.
    pub initrd: Option<(u64, u64)>,
}

/// Build a device tree for the emulated `virt`-style board.
#[must_use]
pub fn virt_dtb(cfg: &DtbConfig) -> Vec<u8> {
    let mut fdt = FdtBuilder::new();

    fdt.begin_node(""); // root
    fdt.prop_u32("#address-cells", 2);
    fdt.prop_u32("#size-cells", 2);
    fdt.prop_strlist("compatible", &["linux,dummy-virt"]);
    fdt.prop_str("model", "v64-virt");
    fdt.prop_u32("interrupt-parent", PHANDLE_GIC);

    // /chosen
    fdt.begin_node("chosen");
    fdt.prop_str("bootargs", cfg.bootargs);
    fdt.prop_str("stdout-path", "/pl011@9000000");
    if let Some((start, end)) = cfg.initrd {
        fdt.prop_u64("linux,initrd-start", start);
        fdt.prop_u64("linux,initrd-end", end);
    }
    fdt.end_node();

    // /memory
    fdt.begin_node("memory@40000000");
    fdt.prop_str("device_type", "memory");
    fdt.prop("reg", &reg_2_2(cfg.mem_base, cfg.mem_size));
    fdt.end_node();

    // /cpus
    fdt.begin_node("cpus");
    fdt.prop_u32("#address-cells", 1);
    fdt.prop_u32("#size-cells", 0);
    fdt.begin_node("cpu@0");
    fdt.prop_str("device_type", "cpu");
    fdt.prop_strlist("compatible", &["arm,cortex-a72"]);
    fdt.prop_u32("reg", 0);
    fdt.prop_str("enable-method", "psci");
    fdt.end_node();
    fdt.end_node();

    // /psci
    fdt.begin_node("psci");
    fdt.prop_strlist("compatible", &["arm,psci-1.0", "arm,psci-0.2"]);
    fdt.prop_str("method", "hvc");
    fdt.end_node();

    // /timer (architected generic timer): secure-phys, phys, virt, hyp PPIs.
    fdt.begin_node("timer");
    fdt.prop_strlist("compatible", &["arm,armv8-timer"]);
    fdt.prop_cells(
        "interrupts",
        &[
            GIC_PPI, 13, IRQ_LEVEL_LOW, // secure physical (IRQ 29)
            GIC_PPI, 14, IRQ_LEVEL_LOW, // non-secure physical (IRQ 30)
            GIC_PPI, 11, IRQ_LEVEL_LOW, // virtual (IRQ 27)
            GIC_PPI, 10, IRQ_LEVEL_LOW, // hypervisor (IRQ 26)
        ],
    );
    fdt.end_node();

    // /intc (GICv2): distributor + CPU interface.
    fdt.begin_node("intc@8000000");
    fdt.prop_strlist("compatible", &["arm,cortex-a15-gic", "arm,gic-400"]);
    fdt.prop_empty("interrupt-controller");
    fdt.prop_u32("#interrupt-cells", 3);
    fdt.prop_u32("#address-cells", 0);
    let mut reg = reg_2_2(cfg.gicd_base, 0x1000);
    reg.extend_from_slice(&reg_2_2(cfg.gicc_base, 0x1000));
    fdt.prop("reg", &reg);
    fdt.prop_u32("phandle", PHANDLE_GIC);
    fdt.end_node();

    // /apb-pclk: fixed clock feeding the UART.
    fdt.begin_node("apb-pclk");
    fdt.prop_strlist("compatible", &["fixed-clock"]);
    fdt.prop_u32("#clock-cells", 0);
    fdt.prop_u32("clock-frequency", 24_000_000);
    fdt.prop_str("clock-output-names", "clk24mhz");
    fdt.prop_u32("phandle", PHANDLE_CLK);
    fdt.end_node();

    // /pl011 UART console.
    fdt.begin_node("pl011@9000000");
    fdt.prop_strlist("compatible", &["arm,pl011", "arm,primecell"]);
    fdt.prop("reg", &reg_2_2(cfg.uart_base, 0x1000));
    fdt.prop_cells("interrupts", &[GIC_SPI, cfg.uart_irq - 32, IRQ_LEVEL_HIGH]);
    fdt.prop_cells("clocks", &[PHANDLE_CLK, PHANDLE_CLK]);
    fdt.prop_strlist("clock-names", &["uartclk", "apb_pclk"]);
    fdt.end_node();

    fdt.end_node(); // root
    fdt.finish()
}

/// A `reg = <base size>` value with `#address-cells = #size-cells = 2`.
fn reg_2_2(base: u64, size: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&base.to_be_bytes());
    v.extend_from_slice(&size.to_be_bytes());
    v
}
