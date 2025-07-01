use crate::error::SrsFromFileError;
use math_lib::{
    cyclic_group::IsGroup,
    errors::DeserializationError,
    traits::{AsBytes, Deserializable},
};

pub struct StructuredReferenceString<G2Point: IsGroup> {
    pub g: G2Point,
}

impl<G2Point: IsGroup> StructuredReferenceString<G2Point> {
    pub fn new(g: G2Point) -> Self {
        Self { g }
    }
}

impl<G2Point: IsGroup + Deserializable> StructuredReferenceString<G2Point> {
    pub fn from_file(file_path: &str) -> Result<Self, SrsFromFileError> {
        let bytes = std::fs::read(file_path)?;
        Ok(Self::deserialize(&bytes)?)
    }
}

impl<G2Point: IsGroup + AsBytes> AsBytes for StructuredReferenceString<G2Point> {
    fn as_bytes(&self) -> Vec<u8> {
        self.g.as_bytes()
    }
}

impl<G2Point: IsGroup + Deserializable> Deserializable for StructuredReferenceString<G2Point> {
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError> {
        let g = G2Point::deserialize(bytes)?;
        Ok(Self { g })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math_lib::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::{
                default_types::FrElement, pairing::BLS12381AtePairing, twist::BLS12381TwistCurve,
            },
            traits::{IsEllipticCurve, IsPairing},
        },
        unsigned_integer::element::U256,
    };
    use rand::Rng;

    fn create_srs() -> StructuredReferenceString<<BLS12381AtePairing as IsPairing>::G2Point> {
        let mut rng = rand::thread_rng();
        let seed = FrElement::new(U256 {
            limbs: [
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ],
        });
        StructuredReferenceString::new(
            BLS12381TwistCurve::generator().operate_with_self(seed.representative()),
        )
    }

    #[test]
    fn test_srs_serialization() {
        let srs = create_srs();
        let bytes = srs.as_bytes();
        let srs2 = StructuredReferenceString::deserialize(&bytes).unwrap();
        assert_eq!(srs.g, srs2.g);
    }
}
