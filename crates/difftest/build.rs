//! When building against the Unicorn oracle, QEMU's 128-bit atomic helpers
//! (e.g. `__atomic_compare_exchange_16`, used by `atomic16_cmpxchg` in
//! `cputlb.c`) resolve to libatomic on x86-64. Unicorn's own build script links
//! pthread and m but not atomic, so we add it here. Gated on the `unicorn`
//! feature so non-oracle builds don't require libatomic.

fn main() {
    if std::env::var_os("CARGO_FEATURE_UNICORN").is_some() {
        println!("cargo:rustc-link-lib=dylib=atomic");
    }
}
