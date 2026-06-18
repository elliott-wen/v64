//! A small direct-mapped translation lookaside buffer (TLB).
//!
//! Stage-1 address translation is the interpreter's hottest operation: every
//! instruction fetch — and every load/store — walks the guest page tables from
//! the root, reading a descriptor from memory at each level. The same handful of
//! pages are walked over and over (a tight loop fetches from one code page
//! thousands of times). This cache remembers the *result* of a walk keyed by the
//! 4KB virtual page, so a repeat access skips the walk entirely.
//!
//! This type is deliberately pure storage with no MMU knowledge: it maps a
//! 4KB-aligned VA to a 4KB-aligned PA plus an opaque `perms` byte that the MMU
//! packs/unpacks (resolved permission bits + the leaf level). The MMU still
//! re-checks permissions against the cached `perms` on every hit, so a cached
//! entry is correct for reads, writes, and fetches at any exception level — only
//! the expensive table walk is elided.
//!
//! Correctness rests on invalidation: the entries become stale when the guest
//! edits a page table, so the MMU [`flush`](Tlb::flush)es the whole cache on any
//! `TLBI` instruction and on writes to the translation control registers
//! (TTBR0/TTBR1/TCR/SCTLR). The architecture *requires* the guest to issue a
//! `TLBI` after changing a translation, so flushing there is sufficient.

/// Number of direct-mapped entries (power of two), indexed by the low bits of
/// the VA page number. Large enough to hold a typical working set of code and
/// data pages without frequent conflict eviction, small enough that a full flush
/// is cheap.
const TLB_SIZE: usize = 1024;

/// Empty-slot sentinel. A real VA page is 4KB-aligned, so its low 12 bits are
/// always zero — `u64::MAX` (all ones) can never collide with a valid tag.
const EMPTY: u64 = u64::MAX;

/// `#[repr(C)]` so generated JIT blocks can read entries directly out of the TLB
/// in shared linear memory (the planned inline-memory fast path): field offsets
/// are pinned and exported as [`ENTRY_TAG`]/[`ENTRY_PA`]/[`ENTRY_PERMS`], stride
/// [`ENTRY_SIZE`], count [`ENTRIES`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Entry {
    /// 4KB-aligned VA this entry maps, or [`EMPTY`].
    tag: u64,
    /// 4KB-aligned PA the VA maps to.
    pa: u64,
    /// MMU-packed resolved permissions + leaf level (opaque here).
    perms: u8,
}

/// Byte offsets and stride of [`Entry`], for the JIT's inline TLB read.
pub const ENTRY_TAG: usize = std::mem::offset_of!(Entry, tag);
pub const ENTRY_PA: usize = std::mem::offset_of!(Entry, pa);
pub const ENTRY_PERMS: usize = std::mem::offset_of!(Entry, perms);
pub const ENTRY_SIZE: usize = std::mem::size_of::<Entry>();
/// Number of direct-mapped entries (power of two; index = `(va>>12) & (ENTRIES-1)`).
pub const ENTRIES: usize = TLB_SIZE;

/// A direct-mapped VA-page → PA-page translation cache. See the module docs.
#[derive(Debug, Clone)]
pub struct Tlb {
    entries: Box<[Entry; TLB_SIZE]>,
}

impl Default for Tlb {
    fn default() -> Self {
        Self::new()
    }
}

impl Tlb {
    #[must_use]
    pub fn new() -> Self {
        Self { entries: Box::new([Entry { tag: EMPTY, pa: 0, perms: 0 }; TLB_SIZE]) }
    }

    /// Map a 4KB-aligned VA to its slot index.
    fn index(va_page: u64) -> usize {
        ((va_page >> 12) as usize) & (TLB_SIZE - 1)
    }

    /// Look up a 4KB-aligned VA. Returns `(pa_page, perms)` on a hit.
    #[must_use]
    pub fn lookup(&self, va_page: u64) -> Option<(u64, u8)> {
        let e = &self.entries[Self::index(va_page)];
        (e.tag == va_page).then_some((e.pa, e.perms))
    }

    /// Cache a successful walk: `va_page` and `pa_page` are 4KB-aligned.
    pub fn insert(&mut self, va_page: u64, pa_page: u64, perms: u8) {
        self.entries[Self::index(va_page)] = Entry { tag: va_page, pa: pa_page, perms };
    }

    /// Drop every entry. Called when a translation may have changed (TLBI, or a
    /// write to a translation control register).
    pub fn flush(&mut self) {
        for e in self.entries.iter_mut() {
            e.tag = EMPTY;
        }
    }
}
