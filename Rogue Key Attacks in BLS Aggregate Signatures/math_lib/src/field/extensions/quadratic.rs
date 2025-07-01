use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::{IsField, IsSubFieldOf};
#[cfg(feature = "lambdaworks-serde-binary")]
use crate::traits::ByteConversion;
use core::fmt::Debug;
use core::marker::PhantomData;

/// A general quadratic extension field over `F`
/// with quadratic non residue `Q::residue()`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadraticExtensionField<F, T>
where
    F: IsField,
    T: HasQuadraticNonResidue<F>,
{
    field: PhantomData<F>,
    non_residue: PhantomData<T>,
}

pub type QuadraticExtensionFieldElement<F, T> = FieldElement<QuadraticExtensionField<F, T>>;

/// Trait to fix a quadratic non residue.
/// Used to construct a quadratic extension field by adding
/// a square root of `residue()`.
pub trait HasQuadraticNonResidue<F: IsField> {
    fn residue() -> FieldElement<F>;
}

impl<F, Q> FieldElement<QuadraticExtensionField<F, Q>>
where
    F: IsField,
    Q: Clone + Debug + HasQuadraticNonResidue<F>,
{
    pub fn conjugate(&self) -> Self {
        let [a, b] = self.value();
        Self::new([a.clone(), -b])
    }
}

#[cfg(feature = "lambdaworks-serde-binary")]
impl<F> ByteConversion for [FieldElement<F>; 2]
where
    F: IsField,
{
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

impl<F, Q> IsField for QuadraticExtensionField<F, Q>
where
    F: IsField,
    Q: Clone + Debug + HasQuadraticNonResidue<F>,
{
    type BaseType = [FieldElement<F>; 2];

    /// Returns the component wise addition of `a` and `b`
    fn add(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        [&a[0] + &b[0], &a[1] + &b[1]]
    }

    /// Returns the multiplication of `a` and `b` using the following
    /// equation:
    /// (a0 + a1 * t) * (b0 + b1 * t) = a0 * b0 + a1 * b1 * Q::residue() + (a0 * b1 + a1 * b0) * t
    /// where `t.pow(2)` equals `Q::residue()`.
    fn mul(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        let q = Q::residue();
        let a0b0 = &a[0] * &b[0];
        let a1b1 = &a[1] * &b[1];
        let z = (&a[0] + &a[1]) * (&b[0] + &b[1]);
        [&a0b0 + &a1b1 * q, z - a0b0 - a1b1]
    }

    fn square(a: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        let [a0, a1] = a;
        let v0 = a0 * a1;
        let c0 = (a0 + a1) * (a0 + Q::residue() * a1) - &v0 - Q::residue() * &v0;
        let c1 = &v0 + &v0;
        [c0, c1]
    }

    /// Returns the component wise subtraction of `a` and `b`
    fn sub(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        [&a[0] - &b[0], &a[1] - &b[1]]
    }

    /// Returns the component wise negation of `a`
    fn neg(a: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        [-&a[0], -&a[1]]
    }

    /// Returns the multiplicative inverse of `a`
    /// This uses the equality `(a0 + a1 * t) * (a0 - a1 * t) = a0.pow(2) - a1.pow(2) * Q::residue()`
    fn inv(a: &[FieldElement<F>; 2]) -> Result<[FieldElement<F>; 2], FieldError> {
        let inv_norm = (a[0].pow(2_u64) - Q::residue() * a[1].pow(2_u64)).inv()?;
        Ok([&a[0] * &inv_norm, -&a[1] * inv_norm])
    }

    /// Returns the division of `a` and `b`
    fn div(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        <Self as IsField>::mul(a, &Self::inv(b).unwrap())
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> [FieldElement<F>; 2] {
        [FieldElement::zero(), FieldElement::zero()]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> [FieldElement<F>; 2] {
        [FieldElement::one(), FieldElement::zero()]
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType {
        [FieldElement::from(x), FieldElement::zero()]
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: [FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        x
    }
}

impl<F, Q> IsSubFieldOf<QuadraticExtensionField<F, Q>> for F
where
    F: IsField,
    Q: Clone + Debug + HasQuadraticNonResidue<F>,
{
    fn mul(
        a: &Self::BaseType,
        b: &<QuadraticExtensionField<F, Q> as IsField>::BaseType,
    ) -> <QuadraticExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::mul(a, b[0].value()));
        let c1 = FieldElement::from_raw(F::mul(a, b[1].value()));
        [c0, c1]
    }

    fn add(
        a: &Self::BaseType,
        b: &<QuadraticExtensionField<F, Q> as IsField>::BaseType,
    ) -> <QuadraticExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::add(a, b[0].value()));
        [c0, b[1].clone()]
    }

    fn div(
        a: &Self::BaseType,
        b: &<QuadraticExtensionField<F, Q> as IsField>::BaseType,
    ) -> <QuadraticExtensionField<F, Q> as IsField>::BaseType {
        let b_inv = <QuadraticExtensionField<F, Q> as IsField>::inv(b).unwrap();
        <Self as IsSubFieldOf<QuadraticExtensionField<F, Q>>>::mul(a, &b_inv)
    }

    fn sub(
        a: &Self::BaseType,
        b: &<QuadraticExtensionField<F, Q> as IsField>::BaseType,
    ) -> <QuadraticExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::sub(a, b[0].value()));
        let c1 = FieldElement::from_raw(F::neg(b[1].value()));
        [c0, c1]
    }

    fn embed(a: Self::BaseType) -> <QuadraticExtensionField<F, Q> as IsField>::BaseType {
        [FieldElement::from_raw(a), FieldElement::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <QuadraticExtensionField<F, Q> as IsField>::BaseType,
    ) -> Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

impl<F: IsField, Q: Clone + Debug + HasQuadraticNonResidue<F>>
    FieldElement<QuadraticExtensionField<F, Q>>
{
}
