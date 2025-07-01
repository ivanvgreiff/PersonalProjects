use math_lib::{cyclic_group::IsGroup, unsigned_integer::traits::IsUnsignedInteger};

pub struct PublicKey<G1Point: IsGroup>(pub G1Point); // G1 is a point on the curve

impl<G1Point: IsGroup> PublicKey<G1Point> {
    pub fn new(pk: G1Point) -> Self {
        Self(pk)
    }
}

pub struct PrivateKey<Z: IsUnsignedInteger>(pub Z);

impl<Z: IsUnsignedInteger> PrivateKey<Z> {
    pub fn new(sk: Z) -> Self {
        Self(sk)
    }
}
