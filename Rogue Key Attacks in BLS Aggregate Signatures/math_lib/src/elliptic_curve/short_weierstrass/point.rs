use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        point::ProjectivePoint,
        traits::{EllipticCurveError, FromAffine, IsEllipticCurve},
    },
    errors::DeserializationError,
    field::element::FieldElement,
    traits::{ByteConversion, Deserializable},
};

use super::traits::IsShortWeierstrass;

#[cfg(feature = "alloc")]
use crate::traits::AsBytes;

#[derive(Clone, Debug)]
pub struct ShortWeierstrassProjectivePoint<E: IsEllipticCurve>(pub ProjectivePoint<E>);

impl<E: IsShortWeierstrass> ShortWeierstrassProjectivePoint<E> {
    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates.
    pub const fn new(value: [FieldElement<E::BaseField>; 3]) -> Self {
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

    pub fn double(&self) -> Self {
        let [px, py, pz] = self.coordinates();

        let px_square = px * px;
        let three_px_square = &px_square + &px_square + &px_square;
        let w = E::a() * pz * pz + three_px_square;
        let w_square = &w * &w;

        let s = py * pz;
        let s_square = &s * &s;
        let s_cube = &s * &s_square;
        let two_s_cube = &s_cube + &s_cube;
        let four_s_cube = &two_s_cube + &two_s_cube;
        let eight_s_cube = &four_s_cube + &four_s_cube;

        let b = px * py * &s;
        let two_b = &b + &b;
        let four_b = &two_b + &two_b;
        let eight_b = &four_b + &four_b;

        let h = &w_square - eight_b;
        let hs = &h * &s;

        let pys_square = py * py * s_square;
        let two_pys_square = &pys_square + &pys_square;
        let four_pys_square = &two_pys_square + &two_pys_square;
        let eight_pys_square = &four_pys_square + &four_pys_square;

        let xp = &hs + &hs;
        let yp = w * (four_b - &h) - eight_pys_square;
        let zp = eight_s_cube;
        Self::new([xp, yp, zp])
    }

    pub fn operate_with_affine(&self, other: &Self) -> Self {
        let [px, py, pz] = self.coordinates();
        let [qx, qy, _qz] = other.coordinates();
        let u = qy * pz;
        let v = qx * pz;

        if self.is_neutral_element() {
            return other.clone();
        }
        if other.is_neutral_element() {
            return self.clone();
        }

        if u == *py {
            if v != *px || *py == FieldElement::zero() {
                return Self::new([
                    FieldElement::zero(),
                    FieldElement::one(),
                    FieldElement::zero(),
                ]);
            } else {
                return self.double();
            }
        }

        let u = &u - py;
        let v = &v - px;
        let vv = &v * &v;
        let uu = &u * &u;
        let vvv = &v * &vv;
        let r = &vv * px;
        let a = &uu * pz - &vvv - &r - &r;

        let x = &v * &a;
        let y = &u * (&r - &a) - &vvv * py;
        let z = &vvv * pz;

        Self::new([x, y, z])
    }
}

impl<E: IsEllipticCurve> PartialEq for ShortWeierstrassProjectivePoint<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<E: IsEllipticCurve> Eq for ShortWeierstrassProjectivePoint<E> {}

impl<E: IsShortWeierstrass> FromAffine<E::BaseField> for ShortWeierstrassProjectivePoint<E> {
    fn from_affine(
        x: FieldElement<E::BaseField>,
        y: FieldElement<E::BaseField>,
    ) -> Result<Self, crate::elliptic_curve::traits::EllipticCurveError> {
        if E::defining_equation(&x, &y) != FieldElement::zero() {
            Err(EllipticCurveError::InvalidPoint)
        } else {
            let coordinates = [x, y, FieldElement::one()];
            Ok(ShortWeierstrassProjectivePoint::new(coordinates))
        }
    }
}

impl<E: IsShortWeierstrass> IsGroup for ShortWeierstrassProjectivePoint<E> {
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
    /// Taken from "Moonmath" (Algorithm 7, page 89)
    fn operate_with(&self, other: &Self) -> Self {
        if other.is_neutral_element() {
            self.clone()
        } else if self.is_neutral_element() {
            other.clone()
        } else {
            let [px, py, pz] = self.coordinates();
            let [qx, qy, qz] = other.coordinates();
            let u1 = qy * pz;
            let u2 = py * qz;
            let v1 = qx * pz;
            let v2 = px * qz;
            if v1 == v2 {
                if u1 != u2 || *py == FieldElement::zero() {
                    Self::neutral_element()
                } else {
                    self.double()
                }
            } else {
                let u = u1 - &u2;
                let v = v1 - &v2;
                let w = pz * qz;

                let u_square = &u * &u;
                let v_square = &v * &v;
                let v_cube = &v * &v_square;
                let v_square_v2 = &v_square * &v2;

                let a = &u_square * &w - &v_cube - (&v_square_v2 + &v_square_v2);

                let xp = &v * &a;
                let yp = u * (&v_square_v2 - a) - &v_cube * u2;
                let zp = &v_cube * w;
                Self::new([xp, yp, zp])
            }
        }
    }

    /// Returns the additive inverse of the projective point `p`
    fn neg(&self) -> Self {
        let [px, py, pz] = self.coordinates();
        Self::new([px.clone(), -py, pz.clone()])
    }
}

#[derive(PartialEq)]
pub enum PointFormat {
    Projective,
    Uncompressed,
    // Compressed,
}

#[derive(PartialEq)]
/// Describes the endianess of the internal types of the types
/// For example, in a field made with limbs of u64
/// this is the endianess of those u64
pub enum Endianness {
    BigEndian,
    LittleEndian,
}

impl<E> ShortWeierstrassProjectivePoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    /// Serialize the points in the given format
    #[cfg(feature = "alloc")]
    pub fn serialize(&self, point_format: PointFormat, endianness: Endianness) -> Vec<u8> {
        // TODO: Add more compact serialization formats
        // Uncompressed affine / Compressed

        let mut bytes: Vec<u8> = Vec::new();
        let x_bytes: Vec<u8>;
        let y_bytes: Vec<u8>;
        let z_bytes: Vec<u8>;

        match point_format {
            PointFormat::Projective => {
                let [x, y, z] = self.coordinates();
                if endianness == Endianness::BigEndian {
                    x_bytes = x.to_bytes_be();
                    y_bytes = y.to_bytes_be();
                    z_bytes = z.to_bytes_be();
                } else {
                    x_bytes = x.to_bytes_le();
                    y_bytes = y.to_bytes_le();
                    z_bytes = z.to_bytes_le();
                }
                bytes.extend(&x_bytes);
                bytes.extend(&y_bytes);
                bytes.extend(&z_bytes);
            }
            PointFormat::Uncompressed => {
                let affine_representation = self.to_affine();
                let [x, y, _z] = affine_representation.coordinates();
                if endianness == Endianness::BigEndian {
                    x_bytes = x.to_bytes_be();
                    y_bytes = y.to_bytes_be();
                } else {
                    x_bytes = x.to_bytes_le();
                    y_bytes = y.to_bytes_le();
                }
                bytes.extend(&x_bytes);
                bytes.extend(&y_bytes);
            }
        }
        bytes
    }

    pub fn deserialize(
        bytes: &[u8],
        point_format: PointFormat,
        endianness: Endianness,
    ) -> Result<Self, DeserializationError> {
        match point_format {
            PointFormat::Projective => {
                if bytes.len() % 3 != 0 {
                    return Err(DeserializationError::InvalidAmountOfBytes);
                }

                let len = bytes.len() / 3;
                let x: FieldElement<E::BaseField>;
                let y: FieldElement<E::BaseField>;
                let z: FieldElement<E::BaseField>;

                if endianness == Endianness::BigEndian {
                    x = ByteConversion::from_bytes_be(&bytes[..len])?;
                    y = ByteConversion::from_bytes_be(&bytes[len..len * 2])?;
                    z = ByteConversion::from_bytes_be(&bytes[len * 2..])?;
                } else {
                    x = ByteConversion::from_bytes_le(&bytes[..len])?;
                    y = ByteConversion::from_bytes_le(&bytes[len..len * 2])?;
                    z = ByteConversion::from_bytes_le(&bytes[len * 2..])?;
                }

                if z == FieldElement::zero() {
                    let point = Self::new([x, y, z]);
                    if point.is_neutral_element() {
                        Ok(point)
                    } else {
                        Err(DeserializationError::FieldFromBytesError)
                    }
                } else if E::defining_equation(&(&x / &z), &(&y / &z)) == FieldElement::zero() {
                    Ok(Self::new([x, y, z]))
                } else {
                    Err(DeserializationError::FieldFromBytesError)
                }
            }
            PointFormat::Uncompressed => {
                if bytes.len() % 2 != 0 {
                    return Err(DeserializationError::InvalidAmountOfBytes);
                }

                let len = bytes.len() / 2;
                let x: FieldElement<E::BaseField>;
                let y: FieldElement<E::BaseField>;

                if endianness == Endianness::BigEndian {
                    x = ByteConversion::from_bytes_be(&bytes[..len])?;
                    y = ByteConversion::from_bytes_be(&bytes[len..])?;
                } else {
                    x = ByteConversion::from_bytes_le(&bytes[..len])?;
                    y = ByteConversion::from_bytes_le(&bytes[len..])?;
                }

                if E::defining_equation(&x, &y) == FieldElement::zero() {
                    Ok(Self::new([x, y, FieldElement::one()]))
                } else {
                    Err(DeserializationError::FieldFromBytesError)
                }
            }
        }
    }
}

#[cfg(feature = "alloc")]
impl<E> AsBytes for ShortWeierstrassProjectivePoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    fn as_bytes(&self) -> Vec<u8> {
        self.serialize(PointFormat::Projective, Endianness::LittleEndian)
    }
}

#[cfg(feature = "alloc")]
impl<E> From<ShortWeierstrassProjectivePoint<E>> for Vec<u8>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    fn from(value: ShortWeierstrassProjectivePoint<E>) -> Self {
        value.as_bytes()
    }
}

impl<E> Deserializable for ShortWeierstrassProjectivePoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        Self::deserialize(bytes, PointFormat::Projective, Endianness::LittleEndian)
    }
}
