//! When building against the Unicorn oracle, QEMU's 128-bit atomic helpers
//! (e.g. `__atomic_compare_exchange_16`, used by `atomic16_cmpxchg` in
//! `cputlb.c`) resolve to libatomic on x86-64 Linux. Unicorn's own build script
//! links pthread and m but not atomic, so we add it here. Gated on the `unicorn`
//! feature so non-oracle builds don't require libatomic, and on Linux because
//! other targets (notably macOS / Apple clang) lower these atomics inline, ship
//! no libatomic, and would fail the final link with `library 'atomic' not found`.

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if std::env::var_os("CARGO_FEATURE_UNICORN").is_some() && target_os == "linux" {
        println!("cargo:rustc-link-lib=dylib=atomic");
    }
}
