#![feature(cfg_target_feature)]
#![feature(inclusive_range_syntax, inclusive_range)]
#![feature(i128_type)]
#![feature(const_fn)]

extern crate rand;
extern crate tuple;
extern crate simd;

pub mod real;
pub mod cast;

pub use real::*;
pub use cast::*;
