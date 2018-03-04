#![feature(core_intrinsics)]
#![feature(cfg_target_feature)]
#![feature(inclusive_range_syntax, inclusive_range)]
#![feature(i128_type)]
#![feature(const_fn)]
#![feature(concat_idents)]
#![cfg_attr(feature="simd", feature(stdsimd))]

extern crate rand;
extern crate tuple;

#[cfg(feature="simd")]
extern crate stdsimd;

macro_rules! first_t {
    ($A:ty, $B:tt) => ($A)
}
macro_rules! first_i {
    ($A:ident, $B:tt) => ($A)
}
macro_rules! first_e {
    ($a:expr, $b:tt) => ($a)
}

pub mod real;
pub mod cast;
#[cfg(feature="simd")]
pub mod simd;

pub use real::Real;
pub use cast::*;
