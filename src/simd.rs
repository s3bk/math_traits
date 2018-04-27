use std::mem::transmute;
use stdsimd::simd::*;
use stdsimd::arch::x86_64::*;
use rand::Rng;
use tuple::*;
use real::Real;
use rand::distributions::{IndependentSample, Range as Uniform};

// highly unsafe macro
macro_rules! call {
    ($pre:ident, $name:ident, $post:ident ( $($arg:expr),* ) ) => (
        unsafe {
            use stdsimd::arch::x86_64::*;
            transmute(concat_idents!($pre, $name, $post)( $( transmute($arg) ),* ))
        }
    )
}

macro_rules! impl_simd {
    ($($simd:ident: $scalar:ident, $bool:ty, $pre:ident ~ $post:ident, $Tuple:ident($($idx:tt)*));*) => ( $(
        impl Real for $simd {
            const PI: Self = $simd::splat(::std::$scalar::consts::PI);
            type Bool = $bool;
            type Scalar = $scalar;
            type Iterator = IntoElements<$Tuple<$(first_t!($scalar, $idx)),*>>;

            #[inline(always)]
            fn splat(s: Self::Scalar) -> Self {
                $simd::splat(s)
            }
            
            #[inline(always)]
            fn values(self) -> Self::Iterator {
                let mut arr = [$(first_e!(0., $idx)),*];
                self.store(&mut arr, 0);
                $Tuple::from(arr).into_elements()
            }

            #[inline(always)]
            fn int(v: i16) -> Self { Self::splat($scalar::from(v)) }
            #[inline]
            fn float(f: f64) -> Self {
                let f = f as $scalar;
                Self::splat($scalar::from(f))
            }
            #[inline]
            fn frac(nom: i16, denom: u16) -> Self {
                Self::splat($scalar::from(nom) / $scalar::from(denom))
            }

            #[inline(always)]
            fn wrap(self, at: Self, span: Self) -> Self {
                Real::select(self - span, self, self.gt(at))
            }
            
            fn uniform01<R: Rng>(rng: &mut R) -> Self {
                let uniform01 = Uniform::new(0., 1.);
                $simd::new($(first_e!(uniform01.ind_sample(rng), $idx)),*)
            }
/*
            #[cfg(target_feature="fma")]
            #[inline]
            fn mul_add(self, b: Self, c: Self) -> Self {
                call!($pre, Fma::mul_add(self, b, c)
            }
*/
            #[inline(always)]
            fn abs(self) -> Self {
                Real::select(-self, self, self.le(Self::splat(0.0)))
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                call!($pre, sqrt, $post (self))
            }

            /*
            #[inline(always)]
            fn sin(self) -> Self { self.sin() }

            #[inline(always)]
            fn cos(self) -> Self { self.cos() }

            #[inline(always)]
            fn exp(self) -> Self { self.exp() }

            #[inline(always)]
            fn ln(self) -> Self { self.ln() }
            */
            #[inline(always)]
            fn floor(self) -> Self {
                call!($pre, floor, $post (self))
            }
            #[inline(always)]
            fn ceil(self) -> Self {
                call!($pre, ceil, $post (self))
            }
            
            #[inline(always)]
            fn min(self, other: Self) -> Self {
                call!($pre, min, $post (self, other))
            }
            #[inline(always)]
            fn max(self, other: Self) -> Self {
                call!($pre, max, $post (self, other))
            }
            #[inline(always)]
            fn lt(self, rhs: Self) -> Self::Bool { $simd::lt(self, rhs) }

            #[inline(always)]
            fn le(self, rhs: Self) -> Self::Bool { $simd::le(self, rhs) }

            #[inline(always)]
            fn gt(self, rhs: Self) -> Self::Bool { $simd::gt(self, rhs) }

            #[inline(always)]
            fn ge(self, rhs: Self) -> Self::Bool { $simd::ge(self, rhs) }

            #[inline(always)]
            fn eq(self, rhs: Self) -> Self::Bool { $simd::eq(self, rhs) }

            #[inline(always)]
            fn select(self, other: Self, cond: Self::Bool) -> Self {
                call!($pre, blendv, $post (self, other, cond))
            }
        }
    )* )
}

#[cfg(target_feature = "mmx")]
impl_simd!(f32x4: f32, i32x4, _mm_ ~ _ps, T4(0 1 2 3));

#[cfg(target_feature = "sse2")]
impl_simd!(f64x2: f64, i64x2, _mm_ ~ _pd, T2(0 1));

#[cfg(target_feature = "avx")]
impl_simd!(
    f32x8: f32, i32x8, _mm256_ ~ _ps, T8(0 1 2 3 4 5 6 7);
    f64x4: f64, i64x4, _mm256_ ~ _pd, T4(0 1 2 3)
);
