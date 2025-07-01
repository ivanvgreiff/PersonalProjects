use bls_lib::srs::StructuredReferenceString;
use math_lib::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            default_types::FrElement, twist::BLS12381TwistCurve,
        },
        traits::IsEllipticCurve,
    },
    traits::AsBytes,
    unsigned_integer::element::U256,
};
use rand::Rng;
use std::io::Write;

fn main() {
    let mut rng = rand::thread_rng();
    let seed = FrElement::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    });
    let srs = StructuredReferenceString::new(
        BLS12381TwistCurve::generator().operate_with_self(seed.representative()),
    );

    let mut file = std::fs::File::create("srs_bls12_381").unwrap();

    file.write_all(&srs.as_bytes()).unwrap();

    println!("SRS has been written to the file 'srs_bls12_381'");
}
