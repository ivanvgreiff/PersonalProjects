use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        point::ProjectivePoint,
        traits::{EllipticCurveError, FromAffine, IsEllipticCurve},
    },
    field::element::FieldElement,
};

use super::traits::IsEdwards;

#[derive(Clone, Debug)]
pub struct EdwardsProjectivePoint<E: IsEllipticCurve>(ProjectivePoint<E>);

impl<E: IsEllipticCurve> EdwardsProjectivePoint<E> {
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

impl<E: IsEllipticCurve> PartialEq for EdwardsProjectivePoint<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<E: IsEdwards> FromAffine<E::BaseField> for EdwardsProjectivePoint<E> {
    fn from_affine(
        x: FieldElement<E::BaseField>,
        y: FieldElement<E::BaseField>,
    ) -> Result<Self, crate::elliptic_curve::traits::EllipticCurveError> {
        if E::defining_equation(&x, &y) != FieldElement::zero() {
            Err(EllipticCurveError::InvalidPoint)
        } else {
            let coordinates = [x, y, FieldElement::one()];
            Ok(EdwardsProjectivePoint::new(coordinates))
        }
    }
}

impl<E: IsEllipticCurve> Eq for EdwardsProjectivePoint<E> {}

impl<E: IsEdwards> IsGroup for EdwardsProjectivePoint<E> {
    /// The point at infinity.
    fn neutral_element() -> Self {
        Self::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::one(),
        ])
    }

    fn is_neutral_element(&self) -> bool {
        let [px, py, pz] = self.coordinates();
        px == &FieldElement::zero() && py == pz
    }

    /// Computes the addition of `self` and `other`.
    /// Taken from "Moonmath" (Eq 5.38, page 97)
    fn operate_with(&self, other: &Self) -> Self {
        // This avoids dropping, which in turn saves us from having to clone the coordinates.
        let (s_affine, o_affine) = (self.to_affine(), other.to_affine());

        let [x1, y1, _] = s_affine.coordinates();
        let [x2, y2, _] = o_affine.coordinates();

        let one = FieldElement::one();
        let (x1y2, y1x2) = (x1 * y2, y1 * x2);
        let (x1x2, y1y2) = (x1 * x2, y1 * y2);
        let dx1x2y1y2 = E::d() * &x1x2 * &y1y2;

        let num_s1 = &x1y2 + &y1x2;
        let den_s1 = &one + &dx1x2y1y2;

        let num_s2 = &y1y2 - E::a() * &x1x2;
        let den_s2 = &one - &dx1x2y1y2;

        Self::new([&num_s1 / &den_s1, &num_s2 / &den_s2, one])
    }

    /// Returns the additive inverse of the projective point `p`
    fn neg(&self) -> Self {
        let [px, py, pz] = self.coordinates();
        Self::new([-px, py.clone(), pz.clone()])
    }
}
