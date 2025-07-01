use crate::{
    elliptic_curve::{
        edwards::{point::EdwardsProjectivePoint, traits::IsEdwards},
        traits::IsEllipticCurve,
    },
    field::{element::FieldElement, fields::p448_goldilocks_prime_field::P448GoldilocksPrimeField},
};

#[derive(Debug, Clone)]
pub struct Ed448Goldilocks;

impl IsEllipticCurve for Ed448Goldilocks {
    type BaseField = P448GoldilocksPrimeField;
    type PointRepresentation = EdwardsProjectivePoint<Self>;

    /// Taken from https://www.rfc-editor.org/rfc/rfc7748#page-6
    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::from_hex("4f1970c66bed0ded221d15a622bf36da9e146570470f1767ea6de324a3d3a46412ae1af72ab66511433b80e18b00938e2626a82bc70cc05e").unwrap(),
            FieldElement::<Self::BaseField>::from_hex("693f46716eb6bc248876203756c9c7624bea73736ca3984087789c1e05a0c2d73ad3ff1ce67c39c4fdbd132c4ed7c8ad9808795bf230fa14").unwrap(),
            FieldElement::one(),
        ])
    }
}

impl IsEdwards for Ed448Goldilocks {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::one()
    }

    fn d() -> FieldElement<Self::BaseField> {
        -FieldElement::from(39081)
    }
}
