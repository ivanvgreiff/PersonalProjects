use crypto_lib::hash::{hash_to_field::hash_to_field, sha3::Sha3Hasher};
use math_lib::{
    cyclic_group::IsGroup, field::fields::montgomery_backed_prime_fields::IsModulus,
    unsigned_integer::element::UnsignedInteger,
};

pub fn hash_to_curve<M: IsModulus<UnsignedInteger<N>> + Clone, const N: usize, G2Point: IsGroup>(
    g: &G2Point,
    message: &[u8],
) -> G2Point {
    let order = M::MODULUS;
    let l = compute_length(order);
    let input = Sha3Hasher::expand_message(message, b"ACDS-BLS", l as u64).unwrap();
    let h = hash_to_field::<M, N>(&input, 1);
    g.operate_with_self(h[0].representative())
}

fn compute_length<const N: usize>(order: UnsignedInteger<N>) -> usize {
    // L = ceil((ceil(log2(p)) + k) / 8), where k is the security parameter of the cryptosystem (e.g. k = ceil(log2(p) / 2))
    let log2_p = order.limbs.len() << 3;
    ((log2_p << 3) + (log2_p << 2)) >> 3
}

#[cfg(test)]
mod tests {
    use crate::hash::hash_to_curve;
    use math_lib::elliptic_curve::{
        short_weierstrass::curves::bls12_381::{curve::BLS12381Curve, default_types::FrConfig},
        traits::IsEllipticCurve,
    };

    #[test]
    fn test_hash_to_bls12381() {
        let message1 = b"Hello, ACDS!";
        let message2 = b"hello, ACDS!";
        let g = BLS12381Curve::generator();
        let hash1 = hash_to_curve::<FrConfig, 4, _>(&g, message1);
        let hash2 = hash_to_curve::<FrConfig, 4, _>(&g, message2);

        assert_ne!(hash1, hash2);
    }
}
