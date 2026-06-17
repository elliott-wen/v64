//! End-to-end: the virtual timer fires (count reaches CVAL) and the resulting
//! PPI is delivered as an IRQ exception through the machine loop.

use std::cell::Cell;
use std::rc::Rc;

use aarch64_cpu_state::CpuState;
use aarch64_decoder::sysreg_key;
use aarch64_interp::{GuestMem, Memory, StopReason};
use aarch64_platform::{Bus, Clock, Gic, Machine};

const GICD: u64 = 0x0800_0000;
const GICC: u64 = 0x0801_0000;

const MAIN: u64 = 0x4000_0000;
const VBAR: u64 = 0x4000_2000;
const IRQ_VECTOR: u64 = VBAR + 0x280; // current-EL/SP_ELx IRQ slot

const PPI_VIRT_TIMER: u32 = 27;
const FREQ: u64 = 62_500_000;

const B_SELF: u32 = 0x1400_0000;

/// A clock whose tick count the test sets explicitly — deterministic, unlike the
/// host clock. (This is also the shape of the future browser `performance.now()`
/// backend: an externally-driven value behind the same trait.)
struct ManualClock(Rc<Cell<u64>>);
impl Clock for ManualClock {
    fn now(&self) -> u64 {
        self.0.get()
    }
}

fn cntv_ctl() -> u32 {
    sysreg_key(3, 3, 14, 3, 1)
}
fn cntv_cval() -> u32 {
    sysreg_key(3, 3, 14, 3, 2)
}
fn vbar_key() -> u32 {
    sysreg_key(3, 0, 12, 0, 0)
}

/// Build a machine with the virtual timer armed (enabled, CVAL = 1000) and the
/// timer PPI enabled at the GIC. `main` and the IRQ vector both spin in place.
/// Returns the machine and the shared clock handle.
fn setup() -> (Machine, Rc<Cell<u64>>) {
    let ticks = Rc::new(Cell::new(0u64));

    let gic = Gic::new();
    let mut bus = Bus::new(Memory::new(MAIN, 0x1_0000));
    bus.map(GICD, 0x10000, Box::new(gic.distributor()));
    bus.map(GICC, 0x10000, Box::new(gic.cpu_interface()));

    // Enable the timer PPI (IRQ 27, word 0 bit 27) and open the controller.
    bus.write_u32(GICD + 0x100, 1 << PPI_VIRT_TIMER);
    bus.write_u32(GICD + 0x000, 1);
    bus.write_u32(GICC + 0x000, 1);
    bus.write_u32(GICC + 0x004, 0xF0);

    // main spins; the IRQ handler spins.
    bus.ram_mut().write(MAIN, &B_SELF.to_le_bytes());
    bus.ram_mut().write(IRQ_VECTOR, &B_SELF.to_le_bytes());

    let mut cpu = CpuState::new(); // EL1h, IRQs unmasked
    cpu.pc = MAIN;
    cpu.sysregs.insert(vbar_key(), VBAR);
    cpu.sysregs.insert(cntv_ctl(), 1); // enabled
    cpu.sysregs.insert(cntv_cval(), 1000); // fire at count 1000

    let m = Machine::with_clock(cpu, bus, gic, Box::new(ManualClock(ticks.clone())), FREQ);
    (m, ticks)
}

#[test]
fn virtual_timer_fires_and_vectors() {
    let (mut m, ticks) = setup();
    m.set_timer_interval(1); // sample every step for a deterministic check

    // Before the deadline: timer idle, main keeps spinning.
    ticks.set(500);
    m.step();
    assert_eq!(m.cpu.pc, MAIN, "count < CVAL: no interrupt");

    // Past the deadline: the timer asserts PPI 27 and we vector to the handler.
    ticks.set(2000);
    m.step();
    assert_eq!(m.cpu.pc, IRQ_VECTOR, "timer IRQ delivered");
}

#[test]
fn clock_sampling_is_throttled_to_the_interval() {
    let (mut m, ticks) = setup();
    let interval = 8;
    m.set_timer_interval(interval);

    m.step(); // first step samples (count 0, below CVAL); window now open
    ticks.set(2000); // deadline crossed, but mid-window the clock isn't re-read

    // The very next step must not observe the new time.
    m.step();
    assert_eq!(m.cpu.pc, MAIN, "throttled: deadline not yet sampled");

    // Within one sampling window the clock is re-read and the timer fires.
    let fired = (0..interval).any(|_| {
        m.step();
        m.cpu.pc == IRQ_VECTOR
    });
    assert!(fired, "timer fires within one sampling interval");
}

const WFI: u32 = 0xd503_207f;
const B_BACK_ONE: u32 = 0x17ff_ffff; // B .-4

/// WFI fast-forward: a guest idling in `WFI; b .-4` must not busy-spin. The
/// machine doesn't block — it returns an idle deadline; the host (here, the
/// test) advances time to it and re-enters. The PPI then fires and the IRQ wakes
/// the guest into its handler. This models the browser driver (no `sleep`).
#[test]
fn wfi_fast_forwards_to_the_timer_deadline() {
    let ticks = Rc::new(Cell::new(0u64));

    let gic = Gic::new();
    let mut bus = Bus::new(Memory::new(MAIN, 0x1_0000));
    bus.map(GICD, 0x10000, Box::new(gic.distributor()));
    bus.map(GICC, 0x10000, Box::new(gic.cpu_interface()));
    bus.write_u32(GICD + 0x100, 1 << PPI_VIRT_TIMER);
    bus.write_u32(GICD + 0x000, 1);
    bus.write_u32(GICC + 0x000, 1);
    bus.write_u32(GICC + 0x004, 0xF0);

    // main: WFI, then branch back to the WFI (a tight idle loop).
    bus.ram_mut().write(MAIN, &WFI.to_le_bytes());
    bus.ram_mut().write(MAIN + 4, &B_BACK_ONE.to_le_bytes());
    bus.ram_mut().write(IRQ_VECTOR, &B_SELF.to_le_bytes()); // handler spins

    let mut cpu = CpuState::new(); // EL1h, IRQs unmasked
    cpu.pc = MAIN;
    cpu.sysregs.insert(vbar_key(), VBAR);
    cpu.sysregs.insert(cntv_ctl(), 1);
    cpu.sysregs.insert(cntv_cval(), 1000); // deadline well within the idle cap

    let mut m = Machine::with_clock(cpu, bus, gic, Box::new(ManualClock(ticks.clone())), FREQ);

    // Drive like the host loop: re-enter run() until the handler is reached,
    // advancing time to the reported idle deadline (the "wait" a real host does).
    let mut woke = false;
    for _ in 0..16 {
        match m.run(IRQ_VECTOR, 10_000_000) {
            StopReason::UntilReached => {
                woke = true;
                break;
            }
            StopReason::CountReached => {
                if let Some(target) = m.idle_until_tick() {
                    ticks.set(target); // host honours the idle deadline
                }
            }
            other => panic!("unexpected stop: {other:?}"),
        }
    }
    assert!(woke, "WFI idle eventually delivered the timer IRQ");
    assert_eq!(m.cpu.pc, IRQ_VECTOR, "woke into the IRQ handler");
    assert!(ticks.get() >= 1000, "clock fast-forwarded to the deadline, not spun");
}
