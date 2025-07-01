use crate::cyclic_group::IsGroup;
use crate::errors::ByteConversionError::{FromBEBytesError, FromLEBytesError};
use crate::errors::CreationError;
use crate::errors::DeserializationError;
use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::{IsFFTField, IsField, IsPrimeField};
use crate::traits::{ByteConversion, Deserializable};
use std::convert::TryInto;

/// Type representing prime fields over unsigned 64-bit integers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct U64PrimeField<const MODULUS: u64>;
pub type U64FieldElement<const MODULUS: u64> = FieldElement<U64PrimeField<MODULUS>>;

pub type F17 = U64PrimeField<17>;
pub type FE17 = U64FieldElement<17>;

impl IsFFTField for F17 {
    const TWO_ADICITY: u64 = 4;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: u64 = 3;
}

impl<const MODULUS: u64> IsField for U64PrimeField<MODULUS> {
    type BaseType = u64;

    fn add(a: &u64, b: &u64) -> u64 {
        ((*a as u128 + *b as u128) % MODULUS as u128) as u64
    }

    fn sub(a: &u64, b: &u64) -> u64 {
        (((*a as u128 + MODULUS as u128) - *b as u128) % MODULUS as u128) as u64
    }

    fn neg(a: &u64) -> u64 {
        MODULUS - a
    }

    fn mul(a: &u64, b: &u64) -> u64 {
        ((*a as u128 * *b as u128) % MODULUS as u128) as u64
    }

    fn div(a: &u64, b: &u64) -> u64 {
        Self::mul(a, &Self::inv(b).unwrap())
    }

    fn inv(a: &u64) -> Result<u64, FieldError> {
        if *a == 0 {
            return Err(FieldError::InvZeroError);
        }
        Ok(Self::pow(a, MODULUS - 2))
    }

    fn eq(a: &u64, b: &u64) -> bool {
        Self::from_u64(*a) == Self::from_u64(*b)
    }

    fn zero() -> u64 {
        0
    }

    fn one() -> u64 {
        1
    }

    fn from_u64(x: u64) -> u64 {
        x % MODULUS
    }

    fn from_base_type(x: u64) -> u64 {
        Self::from_u64(x)
    }
}

impl<const MODULUS: u64> Copy for U64FieldElement<MODULUS> {}

impl<const MODULUS: u64> IsPrimeField for U64PrimeField<MODULUS> {
    type RepresentativeType = u64;

    fn representative(x: &u64) -> u64 {
        *x
    }

    /// Returns how many bits do you need to represent the biggest field element
    /// It expects the MODULUS to be a Prime
    fn field_bit_size() -> usize {
        ((MODULUS - 1).ilog2() + 1) as usize
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError> {
        let mut hex_string = hex_string;
        // Remove 0x if it's on the string
        let mut char_iterator = hex_string.chars();
        if hex_string.len() > 2
            && char_iterator.next().unwrap() == '0'
            && char_iterator.next().unwrap() == 'x'
        {
            hex_string = &hex_string[2..];
        }

        u64::from_str_radix(hex_string, 16).map_err(|_| CreationError::InvalidHexString)
    }

    #[cfg(feature = "std")]
    fn to_hex(x: &u64) -> String {
        format!("{:X}", x)
    }
}

/// Represents an element in Fp. (E.g: 0, 1, 2 are the elements of F3)
impl<const MODULUS: u64> IsGroup for U64FieldElement<MODULUS> {
    fn neutral_element() -> U64FieldElement<MODULUS> {
        U64FieldElement::zero()
    }

    fn operate_with(&self, other: &Self) -> Self {
        *self + *other
    }

    fn neg(&self) -> Self {
        -self
    }
}

impl<const MODULUS: u64> ByteConversion for U64FieldElement<MODULUS> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> Vec<u8> {
        u64::to_be_bytes(*self.value()).into()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> Vec<u8> {
        u64::to_le_bytes(*self.value()).into()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let bytes: [u8; 8] = bytes[0..8].try_into().map_err(|_| FromBEBytesError)?;
        Ok(Self::from(u64::from_be_bytes(bytes)))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let bytes: [u8; 8] = bytes[0..8].try_into().map_err(|_| FromLEBytesError)?;
        Ok(Self::from(u64::from_le_bytes(bytes)))
    }
}

impl<const MODULUS: u64> Deserializable for FieldElement<U64PrimeField<MODULUS>> {
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        Self::from_bytes_be(bytes).map_err(|x| x.into())
    }
}
