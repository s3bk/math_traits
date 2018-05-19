use std::ops::RangeInclusive;

pub trait Cast<O>: Sized {
    /// Try to represent self as O.
    /// If possibe, return Some(Self as O) otherwhise None.
    fn cast(self) -> Option<O>;
    
    /// Try to represent self within the range r of O.
    /// If possibe, return Some(self as O) otherwhise None.
    fn cast_clipped(self, r: RangeInclusive<O>) -> Option<O>;
    
    /// Represent self in the range r
    /// If Self is not in r, choose the nearest end of r.
    /// (returns start <= self as O <= end)
    fn cast_clamped(self, r: RangeInclusive<O>) -> O;
    
    /// Represent self as O
    /// If Self is not in O, choose the nearest O to self.
    /// (returns O::MIN_VALUE <= self as O <= O::MAX_VALUE)
    fn cast_clamping(self) -> O;
}

macro_rules! impl_cast_unchecked {
    ($($src:ty as [$($dst:ty),*],)*) => (
        $( $(
            impl Cast<$dst> for $src {
                #[inline(always)]
                fn cast(self) -> Option<$dst> {
                    Some(self as $dst)
                }
                #[inline(always)]
                fn cast_clipped(self, r: RangeInclusive<$dst>) -> Option<$dst> {
                    let (start, end) = r.into_inner();
                    let v = self as $dst;
                    if v >= start && v <= end {
                        Some(v)
                    } else {
                        None
                    }
                }
                #[inline(always)]
                fn cast_clamped(self, r: RangeInclusive<$dst>) -> $dst {
                    let (start, end) = r.into_inner();
                    let v = self as $dst;
                    if v < start {
                        start
                    } else if v > end {
                        end
                    } else {
                        v
                    }
                }
                #[inline(always)]
                fn cast_clamping(self) -> $dst {
                    self as $dst
                }
            }
        )* )*
    )
}

macro_rules! impl_cast_checked {
    ($($src:ident as [$($dst:ident),*],)*) => (
        $( $(
            impl Cast<$dst> for $src {
                #[inline(always)]
                fn cast(self) -> Option<$dst> {
        		    let min: $src = $dst::min_value() as $src;
                    let max: $src = $dst::max_value() as $src;
                    if self >= min && self <= max {
                        Some(self as $dst)
                    } else {
                        None
                    }
                }
                #[inline(always)]
                fn cast_clipped(self, r: RangeInclusive<$dst>) -> Option<$dst> {
                    let (start, end) = r.into_inner();
                    // if start < 0 (-> big nr), end >= 0, the check fails.
                    // if both < 0, the check fails too
                    // if 0 > start > end, then it passes.
                    if start < end && self >= (start as $src) && self <= (end as $src) {
                        Some(self as $dst)
                    } else {
                        None
                    }
                }
                #[inline(always)]
                fn cast_clamped(self, r: RangeInclusive<$dst>) -> $dst {
                    let (start, end) = r.into_inner();
                    if self >= start as $src {
                        if self <= end as $src {
                            self as $dst
                        } else {
                            end
                        }
                    } else {
                        start
                    }
                }
                #[inline(always)]
                fn cast_clamping(self) -> $dst {
        		    let min: $src = $dst::min_value() as $src;
                    let max: $src = $dst::max_value() as $src;
                    if self < min { $dst::min_value() }
                    else if self > max { $dst::max_value() }
                    else { self as $dst }
                }
            }
        )* )*
    )
}
#[test]
fn test_clip_checked() {
    assert_eq!((-3f32).cast_clipped(0u16..=5), None);
    assert_eq!(8f32.cast_clipped(0u16..=5), None);
    assert_eq!(3f32.cast_clipped(0u16..=5), Some(3));
    assert_eq!(100f32.cast_clipped(0usize..=1000), Some(100));
    assert_eq!(Cast::<u8>::cast_clamping(300i16), 255u8);
}

macro_rules! impl_cast_signed {
    ($( $unsigned:ident, $signed:ident; )*) => (
        $(
            impl Cast<$unsigned> for $signed {
                #[inline(always)]
                fn cast(self) -> Option<$unsigned> {
                    if self >= 0 {
                        Some(self as $unsigned)
                    } else {
                        None
                    }
                }
                #[inline(always)]
                fn cast_clipped(self, r: RangeInclusive<$unsigned>) -> Option<$unsigned> {
                    let (start, end) = r.into_inner();
                    let start = start as $signed;
                    let u = self as $unsigned;
                    if start >= 0 && self >= start && u <= end {
                        Some(u)
                    } else {
                        None
                    }
                }
                #[inline(always)]
                fn cast_clamped(self, r: RangeInclusive<$unsigned>) -> $unsigned {        
                    let (start, end) = r.into_inner();                     
                    let s = start as $signed;
                    if s < 0 || self < s {
                        start
                    } else if self > end as $signed {
                        end
                    } else {
                        self as $unsigned
                    }
                }
                #[inline(always)]
                fn cast_clamping(self) -> $unsigned {
                    if self >= 0 {
                        self as $unsigned
                    } else {
                        0
                    }
                }
            }
            impl Cast<$signed> for $unsigned {
                #[inline(always)]
                fn cast(self) -> Option<$signed> {
                    let s = self as $signed;
                    if s >= 0 {
                        Some(s)
                    } else {
                        None
                    }
                }
                #[inline(always)]
                fn cast_clipped(self, r: RangeInclusive<$signed>) -> Option<$signed> {
                    let (start, end) = r.into_inner();
                    let s = self as $signed;
                    if s >= 0 && s >= start && s <= end {
                        Some(s)
                    } else {
                        None
                    }
                }
                #[inline(always)]
                fn cast_clamped(self, r: RangeInclusive<$signed>) -> $signed {
                    let (start, end) = r.into_inner();
                    let s = self as $signed;
                    if s < 0 || s > end {
                        end
                    } else if s < start {
                        start
                    } else {
                        s
                    }
                }
                #[inline(always)]
                fn cast_clamping(self) -> $signed {
                    let s = self as $signed;
                    if s >= 0 {
                        s
                    } else {
                        $signed::max_value()
                    }
                }
            }
        )*
    )
}
macro_rules! impl_cast_id {
    ($( $a:ty, $b:ty; )*) => (
        $(
            impl Cast<$b> for $a {
                #[inline(always)]
                fn cast(self) -> Option<$b> {
                    Some(self as $b)
                }
                #[inline(always)]
                fn cast_clipped(self, r: RangeInclusive<$b>) -> Option<$b> {
                    let (start, end) = r.into_inner();
                    let b = self as $b;
                    if b >= start && b <= end {
                        Some(b)
                    } else {
                        None
                    }
                }
                #[inline(always)]
                fn cast_clamped(self, r: RangeInclusive<$b>) -> $b {
                    let (start, end) = r.into_inner();
                    let b = self as $b;
                    if b < start {
                        start
                    } else if b > end {
                        end
                    } else {
                        b
                    }
                }
                #[inline(always)]
                fn cast_clamping(self) -> $b {
                    self as $b
                }
                
            }
            impl Cast<$a> for $b {
                #[inline(always)]
                fn cast(self) -> Option<$a> {
                    Some(self as $a)
                }
                #[inline(always)]
                fn cast_clipped(self, r: RangeInclusive<$a>) -> Option<$a> {
                    let (start, end) = r.into_inner();
                    let a = self as $a;
                    if a >= start && a <= end {
                        Some(a)
                    } else {
                        None
                    }
                }
                #[inline(always)]
                fn cast_clamped(self, r: RangeInclusive<$a>) -> $a {
                    let (start, end) = r.into_inner();
                    let a = self as $a;
                    if a < start {
                        start
                    } else if a > end {
                        end
                    } else {
                        a
                    }
                }
                #[inline(always)]
                fn cast_clamping(self) -> $a {
                    self as $a
                }
            }
        )*
    )
}
impl_cast_unchecked!(
     u8 as [    u8, i16, u16, i32, u32, i64, u64, i128, u128, f32, f64, usize, isize],
    u16 as [             u16, i32, u32, i64, u64, i128, u128, f32, f64, usize, isize],
    u32 as [                       u32,      u64, i128, u128, f32, f64              ],
    u64 as [                                 u64,       u128, f32, f64              ],
     i8 as [i8,     i16,      i32,      i64,      i128,       f32, f64,        isize],
    i16 as [        i16,      i32,      i64,      i128, u128, f32, f64,        isize],
    i32 as [                  i32,      i64,      i128, u128, f32, f64              ],
    i64 as [                            i64,      i128, u128, f32, f64              ],
    f32 as [                                                  f32, f64              ],
    f64 as [                                                       f64              ],
);
#[cfg(target_pointer_width = "32")]
impl_cast_unchecked!(
  usize as [                         u64, i128, u128, f32, f64],
  isize as [                    i64,      i128, u128, f32, f64],
);
#[cfg(target_pointer_width = "64")]
impl_cast_unchecked!(
    u32 as [                                                    usize],
  usize as [                                    u128, f32, f64],
  isize as [                              i128, u128, f32, f64],
);
impl_cast_checked!(
    u16 as [u8, i8                                   ],
    i16 as [u8, i8                                   ],
    u32 as [u8, i8, u16, i16                         ],
    i32 as [u8, i8, u16, i16                         ],
    u64 as [u8, i8, u16, i16, u32, i32               ],
    i64 as [u8, i8, u16, i16, u32, i32               ],
    f32 as [u8, i8, u16, i16, u32, i32, u64, i64     , usize, isize],
    f64 as [u8, i8, u16, i16, u32, i32, u64, i64, f32, usize, isize],
);
#[cfg(target_pointer_width = "32")]
impl_cast_checked!(
    u64 as [                                           usize],
  usize as [u8, i8, u16, i16                         ],
  isize as [u8, i8, u16, i16                         ],
);
#[cfg(target_pointer_width = "64")]
impl_cast_checked!(
  usize as [u8, i8, u16, i16, u32, i32               ],
  isize as [u8, i8, u16, i16, u32, i32               ],
);

impl_cast_signed!(u8, i8; u16, i16; u32, i32; u64, i64; usize, isize;);

#[cfg(target_pointer_width = "32")]
impl_cast_id!(usize, u32; isize, i32;);

#[cfg(target_pointer_width = "64")]
impl_cast_id!(usize, u64; isize, i64;);


use tuple::*;
macro_rules! impl_cast {
    ($($Tuple:ident $Arr:ident { $($T:ident . $t:ident . $idx:tt),* } )*) => ($(
        #[allow(non_camel_case_types)]
        impl<$($T, $t),*> Cast<$Tuple<$($t),*>> for $Tuple<$($T),*>
        where $( $T: Cast<$t> ),*
        {
            #[inline(always)]
            fn cast(self) -> Option<$Tuple<$($t),*>> {
                match ( $(self.$idx.cast(), )* ) {
                    ( $( Some($t), )* ) => Some($Tuple($($t),*)),
                    _ => None
                }
            }
            #[inline(always)]
            fn cast_clipped(self, r: RangeInclusive<$Tuple<$($t),*>>) -> Option<$Tuple<$($t),*>> {
                let (start, end) = r.into_inner();
                match ( $(self.$idx.cast_clipped(start.$idx ..= end.$idx), )* ) {
                    ( $( Some($t), )* ) => Some($Tuple($($t),*)),
                    _ => None
                }
            }
            #[inline(always)]
            fn cast_clamped(self, r: RangeInclusive<$Tuple<$($t),*>>) -> $Tuple<$($t),*> {
                let (start, end) = r.into_inner();
                $Tuple( $(self.$idx.cast_clamped(start.$idx ..= end.$idx)),* )
            }
            #[inline(always)]
            fn cast_clamping(self) -> $Tuple<$($t),*> {
                $Tuple( $(self.$idx.cast_clamping()),* )
            }
        }
    )*)
}

impl_tuple!(impl_cast);

trait MinMax {
    fn min_value() -> Self;
    fn max_value() -> Self;
}
impl MinMax for f32 {
    fn min_value() -> f32 { ::std::f32::MIN }
    fn max_value() -> f32 { ::std::f32::MAX }
}
impl MinMax for f64 {
    fn min_value() -> f64 { ::std::f64::MIN }
    fn max_value() -> f64 { ::std::f64::MAX }
}
