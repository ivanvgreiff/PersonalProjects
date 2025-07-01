use crate::hash::hash_to_curve;
use crate::key::{PrivateKey, PublicKey};
use crate::signature::Signature;
use crate::srs::StructuredReferenceString;
use crate::traits::{IsAggregateSignatureScheme, IsSignatureScheme};
use math_lib::{
    cyclic_group::IsGroup, elliptic_curve::traits::IsPairing,
    field::fields::montgomery_backed_prime_fields::IsModulus,
    unsigned_integer::element::UnsignedInteger,
};
use std::marker::PhantomData;

pub struct BonehLynnShacham<M: IsModulus<UnsignedInteger<N>>, const N: usize, P: IsPairing> {
    g1: P::G1Point,
    g2: P::G2Point,
    srs: StructuredReferenceString<P::G2Point>,
    phantom: PhantomData<M>,
}

impl<M, const N: usize, P> BonehLynnShacham<M, N, P>
where
    M: IsModulus<UnsignedInteger<N>>,
    P: IsPairing,
{
    pub fn new(g1: P::G1Point, g2: P::G2Point, srs: StructuredReferenceString<P::G2Point>) -> Self {
        Self {
            g1,
            g2,
            srs,
            phantom: PhantomData,
        }
    }
}

impl<M, const N: usize, P> IsSignatureScheme for BonehLynnShacham<M, N, P>
where
    M: IsModulus<UnsignedInteger<N>> + Clone,
    P: IsPairing,
{
    type PublicKey = PublicKey<P::G1Point>;
    type PrivateKey = PrivateKey<UnsignedInteger<N>>;
    type Signature = Signature<P::G2Point>;

    fn key_gen(&self, sk: &Self::PrivateKey) -> Self::PublicKey {
        Self::PublicKey::new(self.g1.operate_with_self(sk.0))
    }

    fn sign<T: AsRef<[u8]>>(&self, message: T, sk: &Self::PrivateKey) -> Self::Signature {
        let gh = hash_to_curve::<M, N, P::G2Point>(&self.srs.g, message.as_ref());
        Signature::new(gh.operate_with_self(sk.0))
    }

    fn verify<T: AsRef<[u8]>>(
        &self,
        message: T,
        signature: &Self::Signature,
        pk: &Self::PublicKey,
    ) -> bool {
        let gh = hash_to_curve::<M, N, P::G2Point>(&self.srs.g, message.as_ref());
        let left = P::compute(&self.g1, &signature.0).unwrap();
        let right = P::compute(&pk.0, &gh).unwrap();
        left == right
    }
}

impl<M, const N: usize, P> IsAggregateSignatureScheme for BonehLynnShacham<M, N, P>
where
    M: IsModulus<UnsignedInteger<N>> + Clone,
    P: IsPairing,
{
    fn aggregate(
        &self,
        signatures: &[Self::Signature],
        pks: &[Self::PublicKey],
    ) -> Self::Signature {
        let sig = signatures
            .iter()
            .fold(P::G2Point::neutral_element(), |acc, sig| {
                acc.operate_with(&sig.0)
            });
        Signature::new(sig)
    }

    fn verify_aggregate<T: AsRef<[u8]>>(
        &self,
        message: T,
        signature: &Self::Signature,
        pks: &[Self::PublicKey],
    ) -> bool {
        let gh = hash_to_curve::<M, N, P::G2Point>(&self.srs.g, message.as_ref());
        let left = P::compute(&self.g1, &signature.0).unwrap();
        let right = P::compute(
            &pks.iter().fold(P::G1Point::neutral_element(), |acc, pk| {
                acc.operate_with(&pk.0)
            }),
            &gh,
        )
        .unwrap();
        left == right
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math_lib::{
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::{
                curve::BLS12381Curve,
                default_types::{FrConfig, FrElement},
                pairing::BLS12381AtePairing,
                twist::BLS12381TwistCurve,
            },
            traits::IsEllipticCurve,
        },
        unsigned_integer::element::U256,
    };
    use rand::Rng;

    type BLS = BonehLynnShacham<FrConfig, 4, BLS12381AtePairing>;
    type PublicKey = <BLS as IsSignatureScheme>::PublicKey;
    type PrivateKey = <BLS as IsSignatureScheme>::PrivateKey;
    type Signature = <BLS as IsSignatureScheme>::Signature;

    fn rand_keypair(bls: &BLS) -> (PrivateKey, PublicKey) {
        let mut rng = rand::thread_rng();
        let seed = FrElement::new(U256 {
            limbs: [
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ],
        });
        let sk = PrivateKey::new(seed.representative());
        let pk = bls.key_gen(&sk);

        (sk, pk)
    }

    #[test]
    fn test_bls() {
        let bls: BLS = BLS::new(
            BLS12381Curve::generator(),
            BLS12381TwistCurve::generator(),
            StructuredReferenceString::from_file("./setup/srs_bls12_381").unwrap(),
        );
        let (sk, pk) = rand_keypair(&bls);
        let message = "Hello, ACDS!";
        let signature = bls.sign(message, &sk);
        assert!(bls.verify(message, &signature, &pk));
    }

    #[test]
    fn test_bls_aggregate() {
        let bls: BLS = BLS::new(
            BLS12381Curve::generator(),
            BLS12381TwistCurve::generator(),
            StructuredReferenceString::from_file("./setup/srs_bls12_381").unwrap(),
        );
        let (sk1, pk1) = rand_keypair(&bls);
        let (sk2, pk2) = rand_keypair(&bls);
        let message = "Hello, ACDS!";
        let signature1 = bls.sign(message, &sk1);
        let signature2 = bls.sign(message, &sk2);
        let pks = [pk1, pk2];
        let signatures = [signature1, signature2];
        let signature_agg = bls.aggregate(&signatures, &pks);
        assert!(bls.verify_aggregate(message, &signature_agg, &pks));
    }
}
