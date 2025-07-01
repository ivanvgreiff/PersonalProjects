use bls_lib::hash::hash_to_curve;
use bls_lib::srs::StructuredReferenceString;
use bls_lib::{bls::BonehLynnShacham, traits::IsSignatureScheme};
use math_lib::cyclic_group::IsGroup;
use math_lib::elliptic_curve::short_weierstrass::curves::bls12_381::{
    curve::BLS12381Curve,
    default_types::{FrConfig, FrElement},
    pairing::BLS12381AtePairing,
    twist::BLS12381TwistCurve,
};
use math_lib::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use math_lib::elliptic_curve::traits::IsEllipticCurve;
use math_lib::field::fields::montgomery_backed_prime_fields::U64PrimeField;
use math_lib::unsigned_integer::element::U256;
use rand::Rng;

type BLS = BonehLynnShacham<FrConfig, 4, BLS12381AtePairing>;
type PublicKey = <BLS as IsSignatureScheme>::PublicKey;
type PrivateKey = <BLS as IsSignatureScheme>::PrivateKey;
type Signature = <BLS as IsSignatureScheme>::Signature;

/// TODO: Task 2
///
/// Forge an aggregate signature of a foreign public key and a public key of
/// your choice. The function should return a tuple of the forged signature and
/// your chosen public key.
///
/// # Arguments
///
/// * `message` – A vector of bytes representing the message to be signed
/// * `foreign_pk` – A reference to the public key of the foreign party

pub fn attack(message: &[u8], foreign_pk: &PublicKey) -> (PublicKey, Signature) {
    // Get the generator point from the subgroup of our main elliptic curve over F_p
    let g1 = BLS12381Curve::generator();
    let g2 = BLS12381TwistCurve::generator();

    // Lets make a BLM

    // Hash the message to G_2
    let h_m = hash_to_curve::<FrConfig, 4, _>(&g2, message);

    // Compute inverse of the foreign public key in G_1
    let inv_foreign_pk: ShortWeierstrassProjectivePoint<BLS12381Curve> = foreign_pk.0.neg();

    // Compute g_1^beta
    //let mut rng = rand::thread_rng();
    //let beta = FrElement::from(U256::from(rng.gen::<u128>()));

    // let rng = 3 (old)
    //let g1 = &g1.operate_with(&g1);
    //let g_pow_beta = &g1.operate_with(&g1);
    // let rng = 1
    let g_pow_beta = g1;

    // Finally we can compute our rogue key
    let rogue_key = g_pow_beta.operate_with(&inv_foreign_pk);

    // H(m)^beta (old)
    //let h1 = &h_m.operate_with(&h_m);
    //let sig_agg = h1.operate_with(&h1);
    // H(m)^1
    let sig_agg = h_m;

    (PublicKey::new(rogue_key), Signature::new(sig_agg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bls_lib::{srs::StructuredReferenceString, traits::IsAggregateSignatureScheme};

    #[test]
    fn test_attack() {
        // Create a BLS instance using the same parameters used in bls.rs
        let bls = BLS::new(
            BLS12381Curve::generator(),
            BLS12381TwistCurve::generator(),
            StructuredReferenceString::from_file("./setup/srs_bls12_381").unwrap(),
        );

        // Generate a real keypair (foreign key)
        let mut rng = rand::thread_rng();
        let sk_val = FrElement::new(U256 {
            limbs: [
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ],
        });
        let sk = PrivateKey::new(sk_val.representative());
        let pk = bls.key_gen(&sk);

        // Message to be signed
        let message = b"Attack at dawn!";

        // Call the rogue key attack function
        let (rogue_pk, forged_sig) = attack(message, &pk);

        // Construct a list of public keys: [foreign, rogue]
        let pks = vec![pk, rogue_pk];

        // Test that the forged signature verifies as a valid aggregate
        assert!(
            bls.verify_aggregate(message, &forged_sig, &pks),
            "The forged aggregate signature should be valid"
        );
    }
}

