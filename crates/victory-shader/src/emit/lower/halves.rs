//! Console wave64 compute programs on a host whose waves have at most 32
//! lanes (NVIDIA, Intel: `WaveLaneCountMax` 32). Such a host refuses
//! `WaveSize(64)`, which a wave64 program that needs the console's lanes
//! ([`Lower::console_lanes`]) declares, so its pipeline was never made.
//!
//! Each console wave runs as two host waves of 32 lanes ("halves"): the
//! thread relabelling ([`Lower::thread_id`]) gives each host wave a 32-aligned
//! run of flat thread ids, so host wave `2w + h` holds console wave `w`'s
//! lanes `32h..32h + 31`, host lane `n` being console lane `32h + n`. What
//! a lane reads of its own half is exact; reads into the other half have no
//! host form and are refused where they are made:
//!
//! * within a half, exact: DPP (rows of 16, DPP8 groups of 8), `ds_swizzle`
//!   (groups of 32 or quads), and `ds_bpermute_b32`, which in wave64 permutes
//!   each half on its own (RDNA2 ISA, "Data Share Operations"; measured on
//!   RDNA2 in upstream PR #573, banbanzhige's). A lane's own bit of a mask
//!   held as a number (a constant, data loaded or computed by scalar code)
//!   and `v_writelane_b32` (the lane named takes the scalar) are decided by
//!   the console lane ([`Lower::console_lane`]);
//! * across halves, refused: ballots (a mask read as data holds the other
//!   half's lanes), `v_readlane_b32` of a lane that may be in the other half
//!   (SGPR spills read back are not lane reads), and whole-wave reductions
//!   and prefix sums;
//! * `v_readfirstlane_b32` reads the host wave's first lane, as it does in
//!   every program without console lanes (a shortcut the lowering already
//!   takes: the compiler reads first lanes of values the same in all lanes).
//!
//! A refusal in a value no instruction ends up needing costs nothing (the
//! lowering retries with that value missing), so a program is refused only
//! when it really reads across halves.
//!
//! LDS: the console compiler leaves out barriers between the lanes of one
//! wave, and so between all of a group's threads only when the group is one
//! wave. A group of 33 to 64 threads with LDS (two host waves sharing what
//! the console's one wave ordered by itself) is refused; groups of several
//! console waves order their LDS with barriers, and those of 32 threads or
//! fewer are one host wave.
//!
//! Approach after DexNom's fork (DexNom/Victory 5f4c3df, `set_host_max_wave`
//! and the thread relabelling into halves) and upstream PR #2176
//! (n-romanovich: the setting keyed in the shader cache); here the setting
//! is a translation option ([`super::super::Options::host_wave32`]), and
//! cross-half reads are refused where they are lowered instead of by
//! instruction name.

use super::*;

/// Console lanes per host wave in halves.
const HALF: u32 = 32;

impl Lower<'_> {
    /// Whether a wave64 program whose lanes follow the console's runs in
    /// halves: the host's waves are 32 lanes at most.
    pub(super) fn halves_wanted(x: &Parts, needs_wave: bool) -> bool {
        needs_wave && x.native.wave64 && x.host_wave32
    }

    /// Why a program in halves cannot run: LDS a group of one console wave
    /// shares without barriers, split across two host waves.
    pub(super) fn halves_lds(group: u32, lds_dwords: u32) -> Option<Why> {
        (lds_dwords > 0 && group > HALF && group <= 64)
            .then_some("LDS one console wave64 shares without barriers, on a host of 32-lane waves")
    }

    /// The flat thread id, relabelled so that each host wave holds a run of
    /// [`Shader::wave_size`]-aligned ids from its first lane (see
    /// [`Lower::thread_id`]); made once, in the entry block.
    pub(super) fn relabelled_flat(&mut self) -> V {
        if let Some(f) = self.relabelled {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self
            .s
            .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        self.relabelled = Some(f);
        f
    }

    /// This lane's index in its console wave: the host lane, or in halves
    /// the host lane in the half the host wave holds.
    pub(super) fn console_lane(&mut self) -> V {
        if !self.halves {
            return self.lane_index();
        }
        let f = self.relabelled_flat();
        let m = self.s.m.i32(self.wave() - 1);
        self.s.m.bin(self.s.entry, Bin::And, f, m)
    }

    /// Refuses a read into the other half of the console wave in halves.
    pub(super) fn same_half(&self, what: Why) -> Result<(), Why> {
        if self.halves { Err(what) } else { Ok(()) }
    }

    /// The lane mask `ds_bpermute_b32` takes its source lane modulo: in
    /// halves the host wave, which is the half the console permutes within;
    /// else the console wave, as before.
    pub(super) fn permute_lanes(&self) -> u32 {
        if self.halves {
            HALF - 1
        } else {
            self.wave() - 1
        }
    }
}
