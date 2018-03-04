#![feature(core_intrinsics)]
#![feature(cfg_target_feature)]
#![feature(inclusive_range_syntax, inclusive_range)]
#![feature(i128_type)]
#![feature(const_fn)]
#![feature(concat_idents)]
#![cfg_attr(feature="impl_simd", feature(stdsimd))]

extern crate rand;
extern crate tuple;

#[cfg(feature="impl_simd")]
extern crate stdsimd;

pub mod real;
pub mod cast;

pub use real::Real;
pub use cast::*;
