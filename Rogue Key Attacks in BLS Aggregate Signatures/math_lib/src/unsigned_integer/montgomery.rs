use super::element::UnsignedInteger;

pub struct MontgomeryAlgorithms;
impl MontgomeryAlgorithms {
    /// Compute CIOS multiplication of `a` * `b`
    /// `q` is the modulus
    /// `mu` is the inverse of -q modulo 2^{64}
    /// Notice CIOS stands for Coarsely Integrated Operand Scanning
    /// For more information see section 2.3.2 of Tolga Acar's thesis
    /// https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
    #[inline(always)]
    pub const fn cios<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        let mut t = [0_u64; NUM_LIMBS];
        let mut t_extra = [0_u64; 2];
        let mut i: usize = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            // C := 0
            let mut c: u128 = 0;

            // for j=N-1 to 0
            //    (C,t[j]) := t[j] + a[j]*b[i] + C
            let mut cs: u128;
            let mut j: usize = NUM_LIMBS;
            while j > 0 {
                j -= 1;
                cs = t[j] as u128 + (a.limbs[j] as u128) * (b.limbs[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }

            // (t_extra[0],t_extra[1]) := t_extra[1] + C
            cs = (t_extra[1] as u128) + c;
            t_extra[0] = (cs >> 64) as u64;
            t_extra[1] = cs as u64;

            let mut c: u128;

            // m := t[N-1]*q'[N-1] mod D
            let m = ((t[NUM_LIMBS - 1] as u128 * *mu as u128) << 64) >> 64;

            // (C,_) := t[N-1] + m*q[N-1]
            c = (t[NUM_LIMBS - 1] as u128 + m * (q.limbs[NUM_LIMBS - 1] as u128)) >> 64;

            // for j=N-1 to 1
            //    (C,t[j+1]) := t[j] + m*q[j] + C
            let mut j: usize = NUM_LIMBS - 1;
            while j > 0 {
                j -= 1;
                cs = t[j] as u128 + m * (q.limbs[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = ((cs << 64) >> 64) as u64;
            }

            // (C,t[0]) := t_extra[1] + C
            cs = (t_extra[1] as u128) + c;
            c = cs >> 64;
            t[0] = ((cs << 64) >> 64) as u64;

            // t_extra[1] := t_extra[0] + C
            t_extra[1] = t_extra[0] + c as u64;
        }
        let mut result = UnsignedInteger { limbs: t };

        let overflow = t_extra[1] > 0;

        if overflow || UnsignedInteger::const_le(q, &result) {
            (result, _) = UnsignedInteger::sub(&result, q);
        }
        result
    }

    /// Compute CIOS multiplication of `a` * `b`
    /// This is the Algorithm 2 described in the paper
    /// "EdMSM: Multi-Scalar-Multiplication for SNARKs and Faster Montgomery multiplication"
    /// https://eprint.iacr.org/2022/1400.pdf.
    /// It is only suited for moduli with `q[0]` smaller than `2^63 - 1`.
    /// `q` is the modulus
    /// `mu` is the inverse of -q modulo 2^{64}
    #[inline(always)]
    pub fn cios_optimized_for_moduli_with_one_spare_bit<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        let mut t = [0_u64; NUM_LIMBS];
        let mut t_extra;
        let mut i: usize = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            // C := 0
            let mut c: u128 = 0;

            // for j=N-1 to 0
            //    (C,t[j]) := t[j] + a[j]*b[i] + C
            let mut cs: u128;
            let mut j: usize = NUM_LIMBS;
            while j > 0 {
                j -= 1;
                cs = t[j] as u128 + (a.limbs[j] as u128) * (b.limbs[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }

            t_extra = c as u64;

            let mut c: u128;

            // m := t[N-1]*q'[N-1] mod D
            let m = ((t[NUM_LIMBS - 1] as u128 * *mu as u128) << 64) >> 64;

            // (C,_) := t[0] + m*q[0]
            c = (t[NUM_LIMBS - 1] as u128 + m * (q.limbs[NUM_LIMBS - 1] as u128)) >> 64;

            // for j=N-1 to 1
            //    (C,t[j+1]) := t[j] + m*q[j] + C
            let mut j: usize = NUM_LIMBS - 1;
            while j > 0 {
                j -= 1;
                cs = t[j] as u128 + m * (q.limbs[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = ((cs << 64) >> 64) as u64;
            }

            // (C,t[0]) := t_extra + C
            cs = (t_extra as u128) + c;
            t[0] = ((cs << 64) >> 64) as u64;
        }
        let mut result = UnsignedInteger { limbs: t };

        if UnsignedInteger::const_le(q, &result) {
            (result, _) = UnsignedInteger::sub(&result, q);
        }
        result
    }

    // Separated Operand Scanning Method (2.3.1)
    #[inline(always)]
    pub fn sos_square<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        // NOTE: we use explicit `while` loops in this function because profiling pointed
        // at iterators of the form `(<x>..<y>).rev()` as the main performance bottleneck.

        // Step 1: Compute `(hi, lo) = a * a`
        let (mut hi, mut lo) = UnsignedInteger::square(a);

        // Step 2: Add terms to `(hi, lo)` until multiple it
        // is a multiple of both `2^{NUM_LIMBS * 64}` and
        // `q`.
        let mut c: u128 = 0;
        let mut i = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            c = 0;
            let m = (lo.limbs[i] as u128 * *mu as u128) as u64;
            let mut j = NUM_LIMBS;
            while j > 0 {
                j -= 1;
                if i + j >= NUM_LIMBS - 1 {
                    let index = i + j - (NUM_LIMBS - 1);
                    let cs = lo.limbs[index] as u128 + m as u128 * (q.limbs[j] as u128) + c;
                    c = cs >> 64;
                    lo.limbs[index] = cs as u64;
                } else {
                    let index = i + j + 1;
                    let cs = hi.limbs[index] as u128 + m as u128 * (q.limbs[j] as u128) + c;
                    c = cs >> 64;
                    hi.limbs[index] = cs as u64;
                }
            }

            // Carry propagation to `hi`
            let mut t = 0;
            while c > 0 && i >= t {
                let cs = hi.limbs[i - t] as u128 + c;
                c = cs >> 64;
                hi.limbs[i - t] = cs as u64;
                t += 1;
            }
        }

        // Step 3: At this point `overflow * 2^{2 * NUM_LIMBS * 64} + (hi, lo)` is a multiple
        // of `2^{NUM_LIMBS * 64}` and the result is obtained by dividing it by `2^{NUM_LIMBS * 64}`.
        // In other words, `lo` is zero and the result is
        // `overflow * 2^{NUM_LIMBS * 64} + hi`.
        // That number is always strictly smaller than `2 * q`. To normalize it we substract
        // `q` whenever it is larger than `q`.
        // The easy case is when `overflow` is zero. We just use the `sub` function.
        // If `overflow` is 1, then `hi` is smaller than `q`. The function `sub(hi, q)` wraps
        // around `2^{NUM_LIMBS * 64}`. This is the result we need.
        let overflow = c > 0;
        if overflow || UnsignedInteger::const_le(q, &hi) {
            (hi, _) = UnsignedInteger::sub(&hi, q);
        }
        hi
    }
}
