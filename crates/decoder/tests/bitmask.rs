//! DecodeBitMasks tests.

use aarch64_decoder::decode_bit_masks;

#[test]
fn wmask_and_x0_0xfff() {
    // and x0, x0, #0xfff  uses N=1 imms=0b001011 immr=0 -> mask 0xfff
    let (wmask, _) = decode_bit_masks(1, 0b001011, 0, true).unwrap();
    assert_eq!(wmask, 0xfff);
}

#[test]
fn wmask_repeating_32bit() {
    // 32-bit element of 0x0000_00ff replicated -> 0x0000_00ff_0000_00ff
    let (wmask, _) = decode_bit_masks(0, 0b000111, 0, true).unwrap();
    assert_eq!(wmask, 0x0000_00ff_0000_00ff);
}

#[test]
fn reserved_all_ones_run() {
    // imms == levels is reserved for the immediate form.
    assert!(decode_bit_masks(1, 0b111111, 0, true).is_none());
}
