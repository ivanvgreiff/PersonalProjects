use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        point::ProjectivePoint,
        traits::{EllipticCurveError, FromAffine, IsEllipticCurve},
    },
    field::element::FieldElement,
};

use super::traits::IsMontgomery;

#[derive(Clone, Debug)]
pub struct MontgomeryProjectivePoint<E: IsEllipticCurve>(ProjectivePoint<E>);

impl<E: IsEllipticCurve> MontgomeryProjectivePoint<E> {
    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates.
    pub fn new(value: [FieldElement<E::BaseField>; 3]) -> Self {
        Self(ProjectivePoint::new(value))
    }

    /// Returns the `x` coordinate of the point.
    pub fn x(&self) -> &FieldElement<E::BaseField> {
        self.0.x()
    }

    /// Returns the `y` coordinate of the point.
    pub fn y(&self) -> &FieldElement<E::BaseField> {
        self.0.y()
    }

    /// Returns the `z` coordinate of the point.
    pub fn z(&self) -> &FieldElement<E::BaseField> {
        self.0.z()
    }

    /// Returns a tuple [x, y, z] with the coordinates of the point.
    pub fn coordinates(&self) -> &[FieldElement<E::BaseField>; 3] {
        self.0.coordinates()
    }

    /// Creates the same point in affine coordinates. That is,
    /// returns [x / z: y / z: 1] where `self` is [x: y: z].
    /// Panics if `self` is the point at infinity.
    pub fn to_affine(&self) -> Self {
        Self(self.0.to_affine())
    }
}

impl<E: IsEllipticCurve> PartialEq for MontgomeryProjectivePoint<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<E: IsMontgomery> FromAffine<E::BaseField> for MontgomeryProjectivePoint<E> {
    fn from_affine(
        x: FieldElement<E::BaseField>,
        y: FieldElement<E::BaseField>,
    ) -> Result<Self, crate::elliptic_curve::traits::EllipticCurveError> {
        if E::defining_equation(&x, &y) != FieldElement::zero() {
            Err(EllipticCurveError::InvalidPoint)
        } else {
            let coordinates = [x, y, FieldElement::one()];
            Ok(MontgomeryProjectivePoint::new(coordinates))
        }
    }
}

impl<E: IsEllipticCurve> Eq for MontgomeryProjectivePoint<E> {}

impl<E: IsMontgomery> IsGroup for MontgomeryProjectivePoint<E> {
    /// The point at infinity.
    fn neutral_element() -> Self {
        Self::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ])
    }

    fn is_neutral_element(&self) -> bool {
        let pz = self.z();
        pz == &FieldElement::zero()
    }

    /// Computes the addition of `self` and `other`.
    /// Taken from "Moonmath" (Definition 5.2.2.1, page 94)
    fn operate_with(&self, other: &Self) -> Self {
        // One of them is the neutral element.
        if self.is_neutral_element() {
            other.clone()
        } else if other.is_neutral_element() {
            self.clone()
        } else {
            let [x1, y1, _] = self.to_affine().coordinates().clone();
            let [x2, y2, _] = other.to_affine().coordinates().clone();
            // In this case P == -Q
            if x2 == x1 && &y2 + &y1 == FieldElement::zero() {
                Self::neutral_element()
            // The points are the same P == Q
            } else if self == other {
                // P = Q = (x, y)
                // y cant be zero here because if y = 0 then
                // P = Q = (x, 0) and P = -Q, which is the
                // previous case.
                let one = FieldElement::from(1);
                let (a, b) = (E::a(), E::b());

                let x1a = &a * &x1;
                let x1_square = &x1 * &x1;
                let num = &x1_square + &x1_square + x1_square + &x1a + x1a + &one;
                let den = (&b + &b) * &y1;
                let div = num / den;

                let new_x = &div * &div * &b - (&x1 + x2) - a;
                let new_y = div * (x1 - &new_x) - y1;

                Self::new([new_x, new_y, one])
            // In the rest of the cases we have x1 != x2
            } else {
                let num = &y2 - &y1;
                let den = &x2 - &x1;
                let div = num / den;

                let new_x = &div * &div * E::b() - (&x1 + &x2) - E::a();
                let new_y = div * (x1 - &new_x) - y1;

                Self::new([new_x, new_y, FieldElement::one()])
            }
        }
    }

    /// Returns the additive inverse of the projective point `p`
    fn neg(&self) -> Self {
        let [px, py, pz] = self.coordinates();
        Self::new([px.clone(), -py, pz.clone()])
    }
}
