#![feature(core_intrinsics)]
#![feature(const_fn)]
#![feature(concat_idents)]
#![cfg_attr(feature="simd", feature(stdsimd))]

extern crate rand;
extern crate tuple;
#[cfg(feature="simd")]
extern crate packed_simd as simd_;

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
