//! GICv2 interrupt controller: the Distributor (GICD) and CPU Interface (GICC).
//!
//! The Distributor is the global switchboard — per-interrupt enable, pending,
//! active, and priority state, plus the routing decision. The CPU Interface is
//! the per-core front end the handler talks to: it applies the priority mask
//! (`PMR`) and drives the IAR/EOIR acknowledge/deactivate handshake.
//!
//! This is a single-core subset, sufficient to boot Linux on the `virt` machine:
//! one CPU interface, no security/grouping, level/edge config ignored (devices
//! re-assert their line while the condition holds). Interrupt IDs follow the
//! GIC split — SGIs 0..15, PPIs 16..31, SPIs 32.. .
//!
//! Both register blocks share one [`GicInner`] via [`Gic`] (an `Rc<RefCell>`):
//! the bus maps [`Gic::distributor`]/[`Gic::cpu_interface`] as devices, other
//! peripherals raise lines through [`Gic::set_pending`], and the machine loop
//! polls [`Gic::pending_irq`] to decide whether to take an IRQ exception.

use std::cell::RefCell;
use std::rc::Rc;

use crate::MmioDevice;

/// Number of interrupt IDs modelled (32 SGIs/PPIs + SPIs up to 1019, rounded).
const NUM_IRQS: usize = 1024;

/// "No interrupt" / spurious ID returned by IAR when nothing is deliverable.
const SPURIOUS: u32 = 1023;

/// Idle running priority (lower value = higher priority; 0xFF = nothing active).
const IDLE_PRIO: u8 = 0xFF;

/// Shared GICv2 state behind both register blocks.
struct GicInner {
    // --- Distributor ---
    /// GICD_CTLR enable.
    dist_enabled: bool,
    enabled: [bool; NUM_IRQS],
    pending: [bool; NUM_IRQS],
    active: [bool; NUM_IRQS],
    priority: [u8; NUM_IRQS],

    // --- CPU interface ---
    /// GICC_CTLR enable.
    cpu_enabled: bool,
    /// GICC_PMR: interrupts with priority value < pmr are delivered.
    pmr: u8,
    /// Active-priority stack; the top is the current running priority.
    running: Vec<u8>,
    /// Cached highest enabled+pending interrupt (id, priority), regardless of
    /// mask. Recomputed only when enable/pending/priority state changes, so the
    /// per-instruction `signals_irq` check stays O(1) instead of scanning 1024
    /// IDs every step.
    cached_pending: Option<(u32, u8)>,
}

impl GicInner {
    fn new() -> Self {
        Self {
            dist_enabled: false,
            enabled: [false; NUM_IRQS],
            pending: [false; NUM_IRQS],
            active: [false; NUM_IRQS],
            priority: [0; NUM_IRQS],
            cpu_enabled: false,
            pmr: 0,
            running: Vec::new(),
            cached_pending: None,
        }
    }

    fn running_prio(&self) -> u8 {
        self.running.last().copied().unwrap_or(IDLE_PRIO)
    }

    /// Scan for the highest-priority (lowest value, lowest ID on ties)
    /// enabled+pending interrupt. O(NUM_IRQS) — call only on state change.
    fn scan_pending(&self) -> Option<(u32, u8)> {
        let mut best: Option<(u32, u8)> = None;
        for id in 0..NUM_IRQS {
            if self.enabled[id] && self.pending[id] {
                let p = self.priority[id];
                if best.is_none_or(|(_, bp)| p < bp) {
                    best = Some((id as u32, p));
                }
            }
        }
        best
    }

    /// Refresh [`Self::cached_pending`]. Must be called after any change to the
    /// enable/pending/priority arrays.
    fn recompute(&mut self) {
        self.cached_pending = self.scan_pending();
    }

    /// The cached highest enabled+pending interrupt (O(1)).
    fn highest_pending(&self) -> Option<(u32, u8)> {
        self.cached_pending
    }

    /// Would the CPU interface assert IRQ right now?
    fn signals_irq(&self) -> bool {
        if !self.dist_enabled || !self.cpu_enabled {
            return false;
        }
        match self.highest_pending() {
            Some((_, p)) => p < self.pmr && p < self.running_prio(),
            None => false,
        }
    }

    /// GICC_IAR read: acknowledge the highest deliverable interrupt, moving it
    /// pending->active and raising the running priority. Returns its ID, or
    /// `SPURIOUS` if nothing is deliverable.
    fn acknowledge(&mut self) -> u32 {
        match self.highest_pending() {
            Some((id, p)) if p < self.pmr && p < self.running_prio() => {
                let i = id as usize;
                self.pending[i] = false;
                self.active[i] = true;
                self.running.push(p);
                self.recompute(); // pending bit cleared
                id
            }
            _ => SPURIOUS,
        }
    }

    /// GICC_EOIR write: deactivate `id` and drop the running priority.
    fn end_of_interrupt(&mut self, id: u32) {
        if (id as usize) < NUM_IRQS {
            self.active[id as usize] = false;
        }
        self.running.pop();
    }
}

/// A cloneable handle to a shared GICv2. Clones reference the same state.
#[derive(Clone)]
pub struct Gic(Rc<RefCell<GicInner>>);

impl Gic {
    #[must_use]
    pub fn new() -> Self {
        Gic(Rc::new(RefCell::new(GicInner::new())))
    }

    /// Assert interrupt `id` (a peripheral raising its line). Only the rising
    /// edge does work: re-asserting an already-pending line skips the (O(1024))
    /// rescan, which matters because the timer re-drives its PPI every sample.
    pub fn set_pending(&self, id: u32) {
        if (id as usize) < NUM_IRQS {
            let mut g = self.0.borrow_mut();
            if !g.pending[id as usize] {
                g.pending[id as usize] = true;
                g.recompute();
            }
        }
    }

    /// Deassert a pending interrupt that has not yet been acknowledged. Like
    /// [`set_pending`](Self::set_pending), only a real edge triggers a rescan —
    /// the steady state is the timer clearing an already-clear line every sample.
    pub fn clear_pending(&self, id: u32) {
        if (id as usize) < NUM_IRQS {
            let mut g = self.0.borrow_mut();
            if g.pending[id as usize] {
                g.pending[id as usize] = false;
                g.recompute();
            }
        }
    }

    /// Whether the CPU interface is asserting an IRQ to the core right now.
    /// The machine loop ANDs this with `!PSTATE.I` to decide on injection.
    #[must_use]
    pub fn pending_irq(&self) -> bool {
        self.0.borrow().signals_irq()
    }

    /// The Distributor register block, to map on the bus (GICD).
    #[must_use]
    pub fn distributor(&self) -> GicDist {
        GicDist(self.clone())
    }

    /// The CPU Interface register block, to map on the bus (GICC).
    #[must_use]
    pub fn cpu_interface(&self) -> GicCpu {
        GicCpu(self.clone())
    }
}

impl Default for Gic {
    fn default() -> Self {
        Self::new()
    }
}

/// Read/write `size` bytes spanning a 1-bit-per-interrupt bitmap region whose
/// byte 0 corresponds to interrupt `base_irq`. (Each byte covers 8 IRQs.)
fn read_bitmap(arr: &[bool], byte_off: usize, size: u8) -> u64 {
    let base = byte_off * 8;
    let mut v = 0u64;
    for i in 0..(size as usize * 8) {
        if base + i < arr.len() && arr[base + i] {
            v |= 1 << i;
        }
    }
    v
}

/// Apply a write to a bitmap region: `set` chooses set-vs-clear semantics
/// (e.g. ISENABLER vs ICENABLER). Bits written as 0 are no-ops in both.
fn write_bitmap(arr: &mut [bool], byte_off: usize, size: u8, val: u64, set: bool) {
    let base = byte_off * 8;
    for i in 0..(size as usize * 8) {
        if val & (1 << i) != 0 && base + i < arr.len() {
            arr[base + i] = set;
        }
    }
}

/// The Distributor (GICD) register block.
pub struct GicDist(Gic);

impl MmioDevice for GicDist {
    fn name(&self) -> &str {
        "gicd"
    }

    fn read(&mut self, offset: u64, size: u8) -> u64 {
        let g = self.0 .0.borrow();
        let off = offset as usize;
        match off {
            0x000 => u64::from(g.dist_enabled), // GICD_CTLR
            // GICD_TYPER: ITLinesNumber = NUM_IRQS/32 - 1; single CPU.
            0x004 => (NUM_IRQS as u64 / 32) - 1,
            0x008 => 0, // GICD_IIDR
            0x100..0x180 => read_bitmap(&g.enabled, off - 0x100, size), // ISENABLER
            0x180..0x200 => read_bitmap(&g.enabled, off - 0x180, size), // ICENABLER
            0x200..0x280 => read_bitmap(&g.pending, off - 0x200, size), // ISPENDR
            0x280..0x300 => read_bitmap(&g.pending, off - 0x280, size), // ICPENDR
            0x300..0x380 => read_bitmap(&g.active, off - 0x300, size),  // ISACTIVER
            0x380..0x400 => read_bitmap(&g.active, off - 0x380, size),  // ICACTIVER
            // GICD_IPRIORITYR: one byte per interrupt.
            0x400..0x800 => {
                let idx = off - 0x400;
                let mut v = 0u64;
                for i in 0..size as usize {
                    if let Some(p) = g.priority.get(idx + i) {
                        v |= u64::from(*p) << (i * 8);
                    }
                }
                v
            }
            // GICD_ITARGETSR: single CPU -> every interrupt targets CPU0 (0x01).
            0x800..0xC00 => {
                let mut v = 0u64;
                for i in 0..size as usize {
                    v |= 0x01u64 << (i * 8);
                }
                v
            }
            _ => 0, // ICFGR / SGIR / reserved: read as zero
        }
    }

    fn write(&mut self, offset: u64, size: u8, val: u64) {
        let mut g = self.0 .0.borrow_mut();
        let off = offset as usize;
        match off {
            0x000 => g.dist_enabled = val & 1 != 0,
            0x100..0x180 => write_bitmap(&mut g.enabled, off - 0x100, size, val, true),
            0x180..0x200 => write_bitmap(&mut g.enabled, off - 0x180, size, val, false),
            0x200..0x280 => write_bitmap(&mut g.pending, off - 0x200, size, val, true),
            0x280..0x300 => write_bitmap(&mut g.pending, off - 0x280, size, val, false),
            0x300..0x380 => write_bitmap(&mut g.active, off - 0x300, size, val, true),
            0x380..0x400 => write_bitmap(&mut g.active, off - 0x380, size, val, false),
            0x400..0x800 => {
                let idx = off - 0x400;
                for i in 0..size as usize {
                    if let Some(p) = g.priority.get_mut(idx + i) {
                        *p = (val >> (i * 8)) as u8;
                    }
                }
            }
            _ => {} // TYPER/IIDR (RO), ITARGETSR (fixed), ICFGR, SGIR: ignored
        }
        // A guest write may have changed enable/pending/priority; refresh the
        // cache. (Cheap: a write is rare relative to instruction count.)
        g.recompute();
    }
}

/// The CPU Interface (GICC) register block.
pub struct GicCpu(Gic);

impl MmioDevice for GicCpu {
    fn name(&self) -> &str {
        "gicc"
    }

    fn read(&mut self, offset: u64, _size: u8) -> u64 {
        let mut g = self.0 .0.borrow_mut();
        match offset {
            0x00 => u64::from(g.cpu_enabled),    // GICC_CTLR
            0x04 => u64::from(g.pmr),            // GICC_PMR
            0x0C => u64::from(g.acknowledge()),  // GICC_IAR (side-effecting)
            0x14 => u64::from(g.running_prio()), // GICC_RPR
            0x18 => g.highest_pending().map_or(u64::from(SPURIOUS), |(id, _)| u64::from(id)), // HPPIR
            _ => 0, // BPR / IIDR / reserved
        }
    }

    fn write(&mut self, offset: u64, _size: u8, val: u64) {
        let mut g = self.0 .0.borrow_mut();
        match offset {
            0x00 => g.cpu_enabled = val & 1 != 0,
            0x04 => g.pmr = val as u8,
            0x10 => g.end_of_interrupt(val as u32), // GICC_EOIR
            _ => {}                                 // BPR / IAR (RO) / reserved
        }
    }
}
