use crate::errors::CreationError;
use crate::field::errors::FieldError;
use crate::field::traits::{IsField, IsPrimeField};
#[cfg(feature = "lambdaworks-serde-binary")]
use crate::traits::ByteConversion;
use crate::unsigned_integer::element::UnsignedInteger;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P448GoldilocksPrimeField;
pub type U448 = UnsignedInteger<7>;

/// Goldilocks Prime p = 2^448 - 2^224 - 1
pub const P448_GOLDILOCKS_PRIME_FIELD_ORDER: U448 =
    U448::from_hex_unchecked("fffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff");

/// 448-bit unsigned integer represented as
/// a size 8 `u64` array `limbs` of 56-bit words.
/// The least significant word is in the left most position.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct U56x8 {
    limbs: [u64; 8],
}

#[cfg(feature = "lambdaworks-serde-binary")]
impl ByteConversion for U56x8 {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    fn from_bytes_be(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }

    fn from_bytes_le(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}

impl IsField for P448GoldilocksPrimeField {
    type BaseType = U56x8;

    fn add(a: &U56x8, b: &U56x8) -> U56x8 {
        let mut limbs = [0u64; 8];
        for (i, limb) in limbs.iter_mut().enumerate() {
            *limb = a.limbs[i] + b.limbs[i];
        }

        let mut sum = U56x8 { limbs };
        Self::weak_reduce(&mut sum);
        sum
    }

    /// Implements fast Karatsuba Multiplication optimized for the
    /// Godilocks Prime field. Taken from Mike Hamburg's implemenation:
    /// https://sourceforge.net/p/ed448goldilocks/code/ci/master/tree/src/p448/arch_ref64/f_impl.c
    fn mul(a: &U56x8, b: &U56x8) -> U56x8 {
        let (a, b) = (&a.limbs, &b.limbs);
        let mut c = [0u64; 8];

        let mut accum0 = 0u128;
        let mut accum1 = 0u128;
        let mut accum2: u128;

        let mask = (1u64 << 56) - 1;

        let mut aa = [0u64; 4];
        let mut bb = [0u64; 4];
        let mut bbb = [0u64; 4];

        for i in 0..4 {
            aa[i] = a[i] + a[i + 4];
            bb[i] = b[i] + b[i + 4];
            bbb[i] = bb[i] + b[i + 4];
        }

        let widemul = |a: u64, b: u64| -> u128 { (a as u128) * (b as u128) };

        for i in 0..4 {
            accum2 = 0;

            for j in 0..=i {
                accum2 += widemul(a[j], b[i - j]);
                accum1 += widemul(aa[j], bb[i - j]);
                accum0 += widemul(a[j + 4], b[i - j + 4]);
            }
            for j in (i + 1)..4 {
                accum2 += widemul(a[j], b[8 - (j - i)]);
                accum1 += widemul(aa[j], bbb[4 - (j - i)]);
                accum0 += widemul(a[j + 4], bb[4 - (j - i)]);
            }

            accum1 -= accum2;
            accum0 += accum2;

            c[i] = (accum0 as u64) & mask;
            c[i + 4] = (accum1 as u64) & mask;

            accum0 >>= 56;
            accum1 >>= 56;
        }

        accum0 += accum1;
        accum0 += c[4] as u128;
        accum1 += c[0] as u128;
        c[4] = (accum0 as u64) & mask;
        c[0] = (accum1 as u64) & mask;

        accum0 >>= 56;
        accum1 >>= 56;

        c[5] += accum0 as u64;
        c[1] += accum1 as u64;

        U56x8 { limbs: c }
    }

    fn sub(a: &U56x8, b: &U56x8) -> U56x8 {
        let co1 = ((1u64 << 56) - 1) * 2;
        let co2 = co1 - 2;

        let mut limbs = [0u64; 8];
        for (i, limb) in limbs.iter_mut().enumerate() {
            *limb =
                a.limbs[i]
                    .wrapping_sub(b.limbs[i])
                    .wrapping_add(if i == 4 { co2 } else { co1 });
        }

        let mut res = U56x8 { limbs };
        Self::weak_reduce(&mut res);
        res
    }

    fn neg(a: &U56x8) -> U56x8 {
        let zero = Self::zero();
        Self::sub(&zero, a)
    }

    fn inv(a: &U56x8) -> Result<U56x8, FieldError> {
        if *a == Self::zero() {
            return Err(FieldError::InvZeroError);
        }
        Ok(Self::pow(
            a,
            P448_GOLDILOCKS_PRIME_FIELD_ORDER - U448::from_u64(2),
        ))
    }

    fn div(a: &U56x8, b: &U56x8) -> U56x8 {
        let b_inv = Self::inv(b).unwrap();
        Self::mul(a, &b_inv)
    }

    /// Taken from https://sourceforge.net/p/ed448goldilocks/code/ci/master/tree/src/per_field/f_generic.tmpl.c
    fn eq(a: &U56x8, b: &U56x8) -> bool {
        let mut c = Self::sub(a, b);
        Self::strong_reduce(&mut c);
        let mut ret = 0u64;
        for limb in c.limbs.iter() {
            ret |= limb;
        }
        ret == 0
    }

    fn zero() -> U56x8 {
        U56x8 { limbs: [0u64; 8] }
    }

    fn one() -> U56x8 {
        let mut limbs = [0u64; 8];
        limbs[0] = 1;
        U56x8 { limbs }
    }

    fn from_u64(x: u64) -> U56x8 {
        let mut limbs = [0u64; 8];
        limbs[0] = x & ((1u64 << 56) - 1);
        limbs[1] = x >> 56;
        U56x8 { limbs }
    }

    fn from_base_type(x: U56x8) -> U56x8 {
        let mut x = x;
        Self::strong_reduce(&mut x);
        x
    }
}

impl IsPrimeField for P448GoldilocksPrimeField {
    type RepresentativeType = U448;

    fn representative(a: &U56x8) -> U448 {
        let mut a = *a;
        Self::strong_reduce(&mut a);

        let mut r = U448::from_u64(0);
        for i in (0..7).rev() {
            r = r << 56;
            r = r + U448::from_u64(a.limbs[i]);
        }
        r
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError> {
        U56x8::from_hex(hex_string)
    }

    #[cfg(feature = "std")]
    fn to_hex(x: &U56x8) -> String {
        U56x8::to_hex(x)
    }

    fn field_bit_size() -> usize {
        448
    }
}

impl P448GoldilocksPrimeField {
    /// Reduces the value in each limb to less than 2^57 (2^56 + 2^8 - 2 is the largest possible value in a limb after this reduction)
    /// Taken from https://sourceforge.net/p/ed448goldilocks/code/ci/master/tree/src/p448/arch_ref64/f_impl.h
    fn weak_reduce(a: &mut U56x8) {
        let a = &mut a.limbs;

        let mask = (1u64 << 56) - 1;
        let tmp = a[7] >> 56;
        a[4] += tmp;

        for i in (1..8).rev() {
            a[i] = (a[i] & mask) + (a[i - 1] >> 56);
        }

        a[0] = (a[0] & mask) + tmp;
    }

    /// Reduces the number to its canonical form
    /// Taken from https://sourceforge.net/p/ed448goldilocks/code/ci/master/tree/src/per_field/f_generic.tmpl.c
    fn strong_reduce(a: &mut U56x8) {
        P448GoldilocksPrimeField::weak_reduce(a);

        const MODULUS: U56x8 = U56x8 {
            limbs: [
                0xffffffffffffff,
                0xffffffffffffff,
                0xffffffffffffff,
                0xffffffffffffff,
                0xfffffffffffffe,
                0xffffffffffffff,
                0xffffffffffffff,
                0xffffffffffffff,
            ],
        };
        let mask = (1u128 << 56) - 1;

        let mut scarry = 0i128;
        for i in 0..8 {
            scarry = scarry + (a.limbs[i] as i128) - (MODULUS.limbs[i] as i128);
            a.limbs[i] = ((scarry as u128) & mask) as u64;
            scarry >>= 56;
        }

        assert!((scarry as u64) == 0 || (scarry as u64).wrapping_add(1) == 0);

        let scarry_0 = scarry as u64;
        let mut carry = 0u128;

        for i in 0..8 {
            carry = carry + (a.limbs[i] as u128) + ((scarry_0 & MODULUS.limbs[i]) as u128);
            a.limbs[i] = (carry & mask) as u64;
            carry >>= 56;
        }

        assert!((carry as u64).wrapping_add(scarry_0) == 0);
    }
}

impl U56x8 {
    pub const fn from_hex(hex_string: &str) -> Result<Self, CreationError> {
        let mut result = [0u64; 8];
        let mut limb = 0;
        let mut limb_index = 0;
        let mut shift = 0;
        let value = hex_string.as_bytes();
        let mut i: usize = value.len();
        while i > 0 {
            i -= 1;
            limb |= match value[i] {
                c @ b'0'..=b'9' => (c as u64 - '0' as u64) << shift,
                c @ b'a'..=b'f' => (c as u64 - 'a' as u64 + 10) << shift,
                c @ b'A'..=b'F' => (c as u64 - 'A' as u64 + 10) << shift,
                _ => {
                    return Err(CreationError::InvalidHexString);
                }
            };
            shift += 4;
            if shift == 56 && limb_index < 7 {
                result[limb_index] = limb;
                limb = 0;
                limb_index += 1;
                shift = 0;
            }
        }
        result[limb_index] = limb;

        Ok(U56x8 { limbs: result })
    }

    #[cfg(feature = "std")]
    pub fn to_hex(&self) -> String {
        let mut hex_string = String::new();
        for &limb in self.limbs.iter().rev() {
            hex_string.push_str(&format!("{:014X}", limb));
        }
        hex_string.trim_start_matches('0').to_string()
    }
}
