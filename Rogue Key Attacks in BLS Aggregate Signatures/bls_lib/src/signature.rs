use math_lib::cyclic_group::IsGroup;

pub struct Signature<G2Point: IsGroup>(pub G2Point);

impl<G2Point: IsGroup> Signature<G2Point> {
    pub fn new(sig: G2Point) -> Self {
        Self(sig)
    }
}
