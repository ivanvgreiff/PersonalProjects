pub trait IsSignatureScheme {
    type PublicKey;
    type PrivateKey;
    type Signature;

    fn key_gen(&self, sk: &Self::PrivateKey) -> Self::PublicKey;
    fn sign<T: AsRef<[u8]>>(&self, message: T, sk: &Self::PrivateKey) -> Self::Signature;
    fn verify<T: AsRef<[u8]>>(
        &self,
        message: T,
        signature: &Self::Signature,
        pk: &Self::PublicKey,
    ) -> bool;
}

pub trait IsAggregateSignatureScheme: IsSignatureScheme {
    fn aggregate(&self, signatures: &[Self::Signature], pks: &[Self::PublicKey])
        -> Self::Signature;
    fn verify_aggregate<T: AsRef<[u8]>>(
        &self,
        message: T,
        signature: &Self::Signature,
        pks: &[Self::PublicKey],
    ) -> bool;
}
