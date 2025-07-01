use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use core::fmt::Debug;
/// Represents an elliptic curve point using the projective short Weierstrass form:
/// y^2 * z = x^3 + a * x * z^2 + b * z^3,
/// where `x`, `y` and `z` variables are field elements.
#[derive(Debug, Clone)]
pub struct ProjectivePoint<E: IsEllipticCurve> {
    pub value: [FieldElement<E::BaseField>; 3],
}

impl<E: IsEllipticCurve> ProjectivePoint<E> {
    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates.
    pub const fn new(value: [FieldElement<E::BaseField>; 3]) -> Self {
        Self { value }
    }

    /// Returns the `x` coordinate of the point.
    pub fn x(&self) -> &FieldElement<E::BaseField> {
        &self.value[0]
    }

    /// Returns the `y` coordinate of the point.
    pub fn y(&self) -> &FieldElement<E::BaseField> {
        &self.value[1]
    }

    /// Returns the `z` coordinate of the point.
    pub fn z(&self) -> &FieldElement<E::BaseField> {
        &self.value[2]
    }

    /// Returns a tuple [x, y, z] with the coordinates of the point.
    pub fn coordinates(&self) -> &[FieldElement<E::BaseField>; 3] {
        &self.value
    }

    /// Creates the same point in affine coordinates. That is,
    /// returns [x / z: y / z: 1] where `self` is [x: y: z].
    /// Panics if `self` is the point at infinity.
    pub fn to_affine(&self) -> Self {
        let [x, y, z] = self.coordinates();
        // If it's the point at infinite
        if z == &FieldElement::zero() {
            // We make sure all the points in the infinite have the same values
            return Self::new([
                FieldElement::zero(),
                FieldElement::one(),
                FieldElement::zero(),
            ]);
        };
        let inv_z = z.inv().unwrap();
        ProjectivePoint::new([x * &inv_z, y * inv_z, FieldElement::one()])
    }
}

impl<E: IsEllipticCurve> PartialEq for ProjectivePoint<E> {
    fn eq(&self, other: &Self) -> bool {
        let [px, py, pz] = self.coordinates();
        let [qx, qy, qz] = other.coordinates();
        (px * qz == pz * qx) && (py * qz == qy * pz)
    }
}

impl<E: IsEllipticCurve> Eq for ProjectivePoint<E> {}
