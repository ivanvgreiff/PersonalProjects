use crate::field::{
    element::FieldElement,
    errors::FieldError,
    extensions::{
        cubic::{CubicExtensionField, HasCubicNonResidue},
        quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
    traits::IsField,
};

use super::field::Mersenne31Field;

//Note: The inverse calculation in mersenne31/plonky3 differs from the default quadratic extension so I implemented the complex extension.
//////////////////
#[derive(Clone, Debug)]
pub struct Mersenne31Complex;

impl IsField for Mersenne31Complex {
    //Elements represents a[0] = real, a[1] = imaginary
    type BaseType = [FieldElement<Mersenne31Field>; 2];

    /// Returns the component wise addition of `a` and `b`
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    //NOTE: THIS uses Gauss algorithm. Bench this against plonky 3 implementation to see what is faster.
    /// Returns the multiplication of `a` and `b` using the following
    /// equation:
    /// (a0 + a1 * t) * (b0 + b1 * t) = a0 * b0 + a1 * b1 * Self::residue() + (a0 * b1 + a1 * b0) * t
    /// where `t.pow(2)` equals `Q::residue()`.
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = a[0] * b[0];
        let a1b1 = a[1] * b[1];
        let z = (a[0] + a[1]) * (b[0] + b[1]);
        [a0b0 - a1b1, z - a0b0 - a1b1]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        let [a0, a1] = a;
        let v0 = a0 * a1;
        let c0 = (a0 + a1) * (a0 - a1);
        let c1 = v0 + v0;
        [c0, c1]
    }
    /// Returns the component wise subtraction of `a` and `b`
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1]]
    }

    /// Returns the component wise negation of `a`
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-a[0], -a[1]]
    }

    /// Returns the multiplicative inverse of `a`
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let inv_norm = (a[0].pow(2_u64) + a[1].pow(2_u64)).inv()?;
        Ok([a[0] * inv_norm, -a[1] * inv_norm])
    }

    /// Returns the division of `a` and `b`
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        Self::mul(a, &Self::inv(b).unwrap())
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> Self::BaseType {
        [FieldElement::zero(), FieldElement::zero()]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> Self::BaseType {
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
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}

pub type Mersenne31ComplexQuadraticExtensionField =
    QuadraticExtensionField<Mersenne31Field, Mersenne31Complex>;

//TODO: Check this should be for complex and not base field
impl HasQuadraticNonResidue<Mersenne31Complex> for Mersenne31Complex {
    // Verifiable in Sage with
    // ```sage
    // p = 2**31 - 1  # Mersenne31
    // F = GF(p)  # The base field GF(p)
    // R.<x> = F[]  # The polynomial ring over F
    // K.<i> = F.extension(x^2 + 1)  # The complex extension field
    // R2.<y> = K[]
    // f2 = y^2 - i - 2
    // assert f2.is_irreducible()
    // ```
    fn residue() -> FieldElement<Mersenne31Complex> {
        FieldElement::from(&Mersenne31Complex::from_base_type([
            FieldElement::<Mersenne31Field>::from(2),
            FieldElement::<Mersenne31Field>::one(),
        ]))
    }
}

pub type Mersenne31ComplexCubicExtensionField =
    CubicExtensionField<Mersenne31Field, Mersenne31Complex>;

impl HasCubicNonResidue<Mersenne31Complex> for Mersenne31Complex {
    // Verifiable in Sage with
    // ```sage
    // p = 2**31 - 1  # Mersenne31
    // F = GF(p)  # The base field GF(p)
    // R.<x> = F[]  # The polynomial ring over F
    // K.<i> = F.extension(x^2 + 1)  # The complex extension field
    // R2.<y> = K[]
    // f2 = y^3 - 5*i
    // assert f2.is_irreducible()
    // ```
    fn residue() -> FieldElement<Mersenne31Complex> {
        FieldElement::from(&Mersenne31Complex::from_base_type([
            FieldElement::<Mersenne31Field>::zero(),
            FieldElement::<Mersenne31Field>::from(5),
        ]))
    }
}
