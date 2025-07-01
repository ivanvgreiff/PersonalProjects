use super::{
    field_extension::{BLS12381PrimeField, Degree2ExtensionField},
    twist::BLS12381TwistCurve,
};
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::element::U256;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

pub const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

pub type BLS12381FieldElement = FieldElement<BLS12381PrimeField>;
pub type BLS12381TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12381Curve;

impl IsEllipticCurve for BLS12381Curve {
    type BaseField = BLS12381PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new_base("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"),
            FieldElement::<Self::BaseField>::new_base("8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1"),
            FieldElement::one()
        ])
    }
}

impl IsShortWeierstrass for BLS12381Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(4)
    }
}

/// This is equal to the frobenius trace of the BLS12 381 curve minus one or seed value z.
pub const MILLER_LOOP_CONSTANT: u64 = 0xd201000000010000;

/// 𝛽 : primitive cube root of unity of 𝐹ₚ that §satisfies the minimal equation
/// 𝛽² + 𝛽 + 1 = 0 mod 𝑝
pub const CUBE_ROOT_OF_UNITY_G1: BLS12381FieldElement = FieldElement::from_hex_unchecked(
    "5f19672fdf76ce51ba69c6076a0f77eaddb3a93be6f89688de17d813620a00022e01fffffffefffe",
);

/// x-coordinate of 𝜁 ∘ 𝜋_q ∘ 𝜁⁻¹, where 𝜁 is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E
pub const ENDO_U: BLS12381TwistCurveFieldElement =
BLS12381TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked("0"),
    FieldElement::from_hex_unchecked("1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaad")
]);

/// y-coordinate of 𝜁 ∘ 𝜋_q ∘ 𝜁⁻¹, where 𝜁 is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E
pub const ENDO_V: BLS12381TwistCurveFieldElement =
BLS12381TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked("135203e60180a68ee2e9c448d77a2cd91c3dedd930b1cf60ef396489f61eb45e304466cf3e67fa0af1ee7b04121bdea2"),
    FieldElement::from_hex_unchecked("6af0e0437ff400b6831e36d6bd17ffe48395dabc2d3435e77f76e17009241c5ee67992f72ec05f4c81084fbede3cc09")
]);

impl ShortWeierstrassProjectivePoint<BLS12381Curve> {
    /// Returns 𝜙(P) = (𝑥, 𝑦) ⇒ (𝛽𝑥, 𝑦), where 𝛽 is the Cube Root of Unity in the base prime field
    /// https://eprint.iacr.org/2022/352.pdf 2 Preliminaries
    fn phi(&self) -> Self {
        // This clone is unsightly
        let mut a = self.clone();
        a.0.value[0] = a.x() * CUBE_ROOT_OF_UNITY_G1;
        a
    }

    /// 𝜙(P) = −𝑢²P
    /// https://eprint.iacr.org/2022/352.pdf 4.3 Prop. 4
    pub fn is_in_subgroup(&self) -> bool {
        self.operate_with_self(MILLER_LOOP_CONSTANT)
            .operate_with_self(MILLER_LOOP_CONSTANT)
            .neg()
            == self.phi()
    }
}

impl ShortWeierstrassProjectivePoint<BLS12381TwistCurve> {
    /// 𝜓(P) = 𝜁 ∘ 𝜋ₚ ∘ 𝜁⁻¹, where 𝜁 is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E,, 𝜋ₚ is the p-power frobenius endomorphism
    /// and 𝜓 satisifies minmal equation 𝑋² + 𝑡𝑋 + 𝑞 = 𝑂
    /// https://eprint.iacr.org/2022/352.pdf 4.2 (7)
    fn psi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        Self::new([
            x.conjugate() * ENDO_U,
            y.conjugate() * ENDO_V,
            z.conjugate(),
        ])
    }

    /// 𝜓(P) = 𝑢P, where 𝑢 = SEED of the curve
    /// https://eprint.iacr.org/2022/352.pdf 4.2
    pub fn is_in_subgroup(&self) -> bool {
        self.psi() == self.operate_with_self(MILLER_LOOP_CONSTANT).neg()
    }
}
