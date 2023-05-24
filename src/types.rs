use std::{fmt::Debug, ops::Neg};

use num_traits::{Bounded, Float, NumAssignOps, NumCast, PrimInt};

// https://stackoverflow.com/questions/40929867/how-do-you-abstract-generics-in-nested-rust-types
pub trait Num:
    num_traits::Num + NumCast + NumAssignOps + Bounded + PartialOrd + Neg<Output = Self> + Debug + Copy
{
}
impl<T> Num for T where
    T: num_traits::Num
        + NumCast
        + NumAssignOps
        + Bounded
        + PartialOrd
        + Neg<Output = Self>
        + Debug
        + Copy
{
}

pub trait NumInt: PrimInt + Num {}
impl<T> NumInt for T where T: PrimInt + Num {}

pub trait NumFloat: Float + Num {}
impl<T> NumFloat for T where T: Float + Num {}
