//! virtio device family: a DMA-capable peripheral set sharing the virtio-mmio
//! transport and split-virtqueue plumbing. Each device is its own submodule;
//! `queue` holds the ring walking they share.

mod blk;
mod gpu;
mod input;
mod queue;

pub use blk::{VirtioBlk, VirtioBlkMmio};
pub use gpu::{VirtioGpu, VirtioGpuMmio};
pub use input::{InputKind, VirtioInput, VirtioInputMmio};
