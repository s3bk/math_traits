use std::ops::{Add, Sub, Mul, Div};
use cast::Cast;
use rand::{Rng};
use rand::distributions::{IndependentSample, Range as Uniform};
use std::fmt::Debug;

use tuple::*;
//Float + NumCast + SampleRange + PartialOrd + Clone + Add + Debug
pub trait Real:
    Sized + Copy + Debug
  + Mul<Output=Self> + Add<Output=Self> + Sub<Output=Self> + Div<Output=Self>
{
    const PI: Self;
    type Bool;
    type Scalar;
    type Iterator: Iterator<Item=Self::Scalar>;

    fn values(self) -> Self::Iterator;
    
    fn int(v: i16) -> Self;
    fn float(f: f64) -> Self;
    
    fn frac(nom: i16, denom: u16) -> Self;
    #[inline]
    fn inv(self) -> Self {
        <Self as Real>::int(1) / self
    }
    
    fn uniform01<R: Rng>(rng: &mut R) -> Self;

    /// |x|
    fn abs(self) -> Self;

    /// sqrt(x)
    fn sqrt(self) -> Self { unimplemented!() }
    
    fn pow(self) -> Self { unimplemented!() }
    
    /* TODO: needs simd impl
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    */

    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    
    /// self * b + c
    #[inline]
    fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }
    
    /// if self exeeds at, subtract span
    fn wrap(self, at: Self, span: Self) -> Self;
    
    fn splat(s: Self::Scalar) -> Self;

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        let clamped_low = min.select(self, self.lt(min));
        max.select(clamped_low, self.gt(max))
    }
    
    fn lt(self, rhs: Self) -> Self::Bool;
    fn le(self, rhs: Self) -> Self::Bool;
    fn gt(self, rhs: Self) -> Self::Bool;
    fn ge(self, rhs: Self) -> Self::Bool;
    fn eq(self, rhs: Self) -> Self::Bool;

    // if cont true. then select self, otherwhise other
    fn select(self, other: Self, cond: Self::Bool) -> Self;
    #[inline]
    fn max(self, other: Self) -> Self {
        self.select(other, self.gt(other))
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        self.select(other, self.lt(other))
    }
}

macro_rules! impl_real {
    ($($t:ident : $fma:ident),*) => ( $(
        impl Real for $t {
            const PI: Self = ::std::$t::consts::PI;
            type Bool = bool;
            type Scalar = $t;
            type Iterator = ::std::iter::Once<$t>;
            
            #[inline(always)]
            fn splat(s: Self::Scalar) -> Self {
                s
            }

            #[inline(always)]
            fn values(self) -> Self::Iterator {
                ::std::iter::once(self)
            }
            
            #[inline(always)]
            fn int(v: i16) -> Self { v.into() }
            
            #[inline(always)]
            fn float(f: f64) -> Self { f.cast().unwrap() }

            #[inline]
            fn frac(nom: i16, denom: u16) -> Self {
                $t::from(nom) / $t::from(denom)
            }
            #[inline]
            fn wrap(self, at: Self, span: Self) -> Self {
                if self > at { self - span } else { self }
            }
            #[inline]
            fn uniform01<R: Rng>(rng: &mut R) -> Self {
                let uniform01 = Uniform::new(0., 1.);
                uniform01.ind_sample(rng)
            }

            #[cfg(target_feature="fma")]
            #[inline]
            fn mul_add(self, b: Self, c: Self) -> Self {
                unsafe { $fma(self, b, c) }
            }
            
            #[inline(always)]
            fn abs(self) -> Self { self.abs() }

            #[inline(always)]
            fn sqrt(self) -> Self { self.sqrt() }

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
                $t::floor(self)
            }
            #[inline(always)]
            fn ceil(self) -> Self {
                $t::ceil(self)
            }
            
            #[inline(always)]
            fn lt(self, rhs: Self) -> Self::Bool { self < rhs }

            #[inline(always)]
            fn le(self, rhs: Self) -> Self::Bool { self <= rhs }

            #[inline(always)]
            fn gt(self, rhs: Self) -> Self::Bool { self > rhs }

            #[inline(always)]
            fn ge(self, rhs: Self) -> Self::Bool { self >= rhs }
            
            #[inline(always)]
            fn eq(self, rhs: Self) -> Self::Bool { self == rhs }

            #[inline(always)]
            fn select(self, other: Self, cond: Self::Bool) -> Self {
                if cond { self } else { other }
            }
        }
    )* )
}

impl_real!(f32: fmaf32, f64: fmaf64);

macro_rules! tuple_init {
    ($($Tuple:ident { $($T:ident . $t:ident . $idx:tt),* } )*) => ($(
    
        impl<T: Real> Real for $Tuple<$(first_i!(T, $T),)*>
        {
            const PI: Self = $Tuple( $(first_e!(T::PI, $T),)* );
            type Bool = $Tuple<$(first_t!(T::Bool, $T)),*>;
            type Scalar = T;
            type Iterator = IntoElements<Self>;

            #[inline]
            fn splat(s: Self::Scalar) -> Self {
                $Tuple( $(first_e!(s, $idx),)* )
            }
            #[inline]
            fn values(self) -> Self::Iterator {
                self.into_elements()
            }

            #[inline]
            fn int(v: i16) -> Self {
                $Tuple( $(first_e!(T::int(v), $idx),)* )
            }
            #[inline]
            fn float(f: f64) -> Self {
                $Tuple( $(first_e!(T::float(f), $idx),)* )
            }
            #[inline]
            fn frac(nom: i16, denom: u16) -> Self {
                $Tuple( $(first_e!(T::frac(nom, denom), $idx),)* )
            }
            
            #[inline]
            fn uniform01<R: Rng>(rng: &mut R) -> Self {
                $Tuple( $(first_e!(T::uniform01(rng), $idx),)* )
            }

            #[inline]
            fn abs(self) -> Self {
                $Tuple( $(T::abs(self.$idx)),* )
            }

            #[inline]
            fn sqrt(self) -> Self {
                $Tuple( $(T::sqrt(self.$idx)),* )
            }

            #[inline]
            fn wrap(self, at: Self, span: Self) -> Self {
                $Tuple( $(T::wrap(self.$idx, at.$idx, span.$idx),)* )
            }

            #[inline]
            fn clamp(self, min: Self, max: Self) -> Self {
                $Tuple( $(T::clamp(self.$idx, min.$idx, max.$idx),)* )
            }

            #[inline]
            fn floor(self) -> Self {
                $Tuple( $(T::floor(self.$idx),)* )
            }

            #[inline]
            fn ceil(self) -> Self {
                $Tuple( $(T::ceil(self.$idx),)* )
            }

            #[inline]
            fn lt(self, rhs: Self) -> Self::Bool {
                $Tuple( $(T::lt(self.$idx, rhs.$idx),)* )
            }
            #[inline]
            fn le(self, rhs: Self) -> Self::Bool {
                $Tuple( $(T::le(self.$idx, rhs.$idx),)* )
            }
            #[inline]
            fn gt(self, rhs: Self) -> Self::Bool {
                $Tuple( $(T::gt(self.$idx, rhs.$idx),)* )
            }
            #[inline]
            fn ge(self, rhs: Self) -> Self::Bool {
                $Tuple( $(T::ge(self.$idx, rhs.$idx),)* )
            }
            #[inline]
            fn eq(self, rhs: Self) -> Self::Bool {
                $Tuple( $(T::eq(self.$idx, rhs.$idx),)* )
            }

            #[inline]
            fn select(self, other: Self, cond: Self::Bool) -> Self {
                $Tuple( $(T::select(self.$idx, other.$idx, cond.$idx),)* )
            }
        }
    )*)
}
impl_tuple!(tuple_init);
