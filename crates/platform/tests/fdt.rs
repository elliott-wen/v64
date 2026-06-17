//! Validate the generated DTB: header fields, a balanced token walk, and that
//! known properties are present with the right values.

use aarch64_platform::{virt_dtb, DtbConfig};

const FDT_BEGIN_NODE: u32 = 1;
const FDT_END_NODE: u32 = 2;
const FDT_PROP: u32 = 3;
const FDT_NOP: u32 = 4;
const FDT_END: u32 = 9;

fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes(b[off..off + 4].try_into().unwrap())
}

fn sample_dtb() -> Vec<u8> {
    virt_dtb(&DtbConfig {
        mem_base: 0x4000_0000,
        mem_size: 0x800_0000,
        gicd_base: 0x0800_0000,
        gicc_base: 0x0801_0000,
        uart_base: 0x0900_0000,
        uart_irq: 33,
        bootargs: "console=ttyAMA0",
        initrd: Some((0x4800_0000, 0x4810_0000)),
        virtio: &[(0x0a00_0000, 48)],
    })
}

#[test]
fn header_is_well_formed() {
    let dtb = sample_dtb();
    assert_eq!(be32(&dtb, 0), 0xd00d_feed, "magic");
    assert_eq!(be32(&dtb, 4) as usize, dtb.len(), "totalsize matches blob");
    assert_eq!(be32(&dtb, 20), 17, "version");
    assert_eq!(be32(&dtb, 24), 16, "last_comp_version");
    // The structure block must be 4-byte aligned (its tokens are u32).
    assert_eq!(be32(&dtb, 8) % 4, 0, "off_dt_struct aligned");
    assert!(be32(&dtb, 8) < be32(&dtb, 12), "struct precedes strings");
}

/// Walk the structure block, returning (final depth, every (name,value) prop).
fn walk(dtb: &[u8]) -> (i32, Vec<(String, Vec<u8>)>) {
    let off_struct = be32(dtb, 8) as usize;
    let off_strings = be32(dtb, 12) as usize;
    let mut pos = off_struct;
    let mut depth = 0i32;
    let mut props = Vec::new();
    loop {
        let tok = be32(dtb, pos);
        pos += 4;
        match tok {
            FDT_BEGIN_NODE => {
                depth += 1;
                // name: null-terminated, padded to 4.
                let start = pos;
                while dtb[pos] != 0 {
                    pos += 1;
                }
                let _ = start;
                pos += 1;
                pos = (pos + 3) & !3;
            }
            FDT_END_NODE => depth -= 1,
            FDT_PROP => {
                let len = be32(dtb, pos) as usize;
                let nameoff = be32(dtb, pos + 4) as usize;
                pos += 8;
                let value = dtb[pos..pos + len].to_vec();
                pos = (pos + len + 3) & !3;
                // Property name is a C-string in the strings block.
                let nstart = off_strings + nameoff;
                let nend = (nstart..dtb.len()).find(|&i| dtb[i] == 0).unwrap();
                let name = String::from_utf8(dtb[nstart..nend].to_vec()).unwrap();
                props.push((name, value));
            }
            FDT_NOP => {}
            FDT_END => break,
            other => panic!("bad token {other:#x} at {pos:#x}"),
        }
    }
    (depth, props)
}

#[test]
fn structure_is_balanced_and_terminated() {
    let dtb = sample_dtb();
    let (depth, _props) = walk(&dtb); // panics on a bad token / unterminated
    assert_eq!(depth, 0, "every begin-node has a matching end-node");
}

#[test]
fn carries_expected_properties() {
    let dtb = sample_dtb();
    let (_, props) = walk(&dtb);
    let find = |n: &str| props.iter().find(|(name, _)| name == n).map(|(_, v)| v.clone());

    assert_eq!(find("bootargs").unwrap(), b"console=ttyAMA0\0");
    assert_eq!(find("stdout-path").unwrap(), b"/pl011@9000000\0");
    assert_eq!(find("method").unwrap(), b"hvc\0", "PSCI conduit");
    // initrd-start is a big-endian u64 at 0x4800_0000.
    assert_eq!(find("linux,initrd-start").unwrap(), 0x4800_0000u64.to_be_bytes());
    // The PL011 compatible string list must include arm,pl011.
    let compat = props.iter().filter(|(n, _)| n == "compatible").any(|(_, v)| {
        v.split(|&b| b == 0).any(|s| s == b"arm,pl011")
    });
    assert!(compat, "pl011 compatible present");
}
