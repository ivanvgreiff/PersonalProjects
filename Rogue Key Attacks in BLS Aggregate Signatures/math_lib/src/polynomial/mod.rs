use super::field::element::FieldElement;
use crate::field::traits::{IsField, IsSubFieldOf};
use std::{fmt::Display, ops};
mod error;

/// Represents the polynomial c_0 + c_1 * X + c_2 * X^2 + ... + c_n * X^n
/// as a vector of coefficients `[c_0, c_1, ... , c_n]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial<FE> {
    pub coefficients: Vec<FE>,
}

impl<F: IsField> Polynomial<FieldElement<F>> {
    /// Creates a new polynomial with the given coefficients
    pub fn new(coefficients: &[FieldElement<F>]) -> Self {
        // Removes trailing zero coefficients at the end
        let mut unpadded_coefficients = coefficients
            .iter()
            .rev()
            .skip_while(|x| **x == FieldElement::zero())
            .cloned()
            .collect::<Vec<FieldElement<F>>>();
        unpadded_coefficients.reverse();
        Polynomial {
            coefficients: unpadded_coefficients,
        }
    }

    pub fn new_monomial(coefficient: FieldElement<F>, degree: usize) -> Self {
        let mut coefficients = vec![FieldElement::zero(); degree];
        coefficients.push(coefficient);
        Self::new(&coefficients)
    }

    pub fn zero() -> Self {
        Self::new(&[])
    }

    pub fn evaluate<E>(&self, x: &FieldElement<E>) -> FieldElement<E>
    where
        E: IsField,
        F: IsSubFieldOf<E>,
    {
        self.coefficients
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| {
                coeff + acc * x.to_owned()
            })
    }

    pub fn evaluate_slice(&self, input: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        input.iter().map(|x| self.evaluate(x)).collect()
    }

    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    pub fn leading_coefficient(&self) -> FieldElement<F> {
        if let Some(coefficient) = self.coefficients.last() {
            coefficient.clone()
        } else {
            FieldElement::zero()
        }
    }

    /// Returns coefficients of the polynomial as an array
    /// \[c_0, c_1, c_2, ..., c_n\]
    /// that represents the polynomial
    /// c_0 + c_1 * X + c_2 * X^2 + ... + c_n * X^n
    pub fn coefficients(&self) -> &[FieldElement<F>] {
        &self.coefficients
    }

    pub fn coeff_len(&self) -> usize {
        self.coefficients().len()
    }

    /// Computes quotient with `x - b` in place.
    pub fn ruffini_division_inplace(&mut self, b: &FieldElement<F>) {
        let mut c = FieldElement::zero();
        for coeff in self.coefficients.iter_mut().rev() {
            *coeff = &*coeff + b * &c;
            std::mem::swap(coeff, &mut c);
        }
        self.coefficients.pop();
    }

    pub fn ruffini_division<L>(&self, b: &FieldElement<L>) -> Polynomial<FieldElement<L>>
    where
        L: IsField,
        F: IsSubFieldOf<L>,
    {
        if let Some(c) = self.coefficients.last() {
            let mut c = c.clone().to_extension();
            let mut coefficients = Vec::with_capacity(self.degree());
            for coeff in self.coefficients.iter().rev().skip(1) {
                coefficients.push(c.clone());
                c = coeff + c * b;
            }
            coefficients = coefficients.into_iter().rev().collect();
            Polynomial::new(&coefficients)
        } else {
            Polynomial::zero()
        }
    }

    /// Computes quotient and remainder of polynomial division.
    ///
    /// Output: (quotient, remainder)
    pub fn long_division_with_remainder(self, dividend: &Self) -> (Self, Self) {
        if dividend.degree() > self.degree() {
            (Polynomial::zero(), self)
        } else {
            let mut n = self;
            let mut q: Vec<FieldElement<F>> = vec![FieldElement::zero(); n.degree() + 1];
            let denominator = dividend.leading_coefficient().inv().unwrap();
            while n != Polynomial::zero() && n.degree() >= dividend.degree() {
                let new_coefficient = n.leading_coefficient() * &denominator;
                q[n.degree() - dividend.degree()] = new_coefficient.clone();
                let d = dividend.mul_with_ref(&Polynomial::new_monomial(
                    new_coefficient,
                    n.degree() - dividend.degree(),
                ));
                n = n - d;
            }
            (Polynomial::new(&q), n)
        }
    }

    pub fn div_with_ref(self, dividend: &Self) -> Self {
        let (quotient, _remainder) = self.long_division_with_remainder(dividend);
        quotient
    }

    pub fn mul_with_ref(&self, factor: &Self) -> Self {
        let degree = self.degree() + factor.degree();
        let mut coefficients = vec![FieldElement::zero(); degree + 1];

        if self.coefficients.is_empty() || factor.coefficients.is_empty() {
            Polynomial::new(&[FieldElement::zero()])
        } else {
            for i in 0..=factor.degree() {
                for j in 0..=self.degree() {
                    coefficients[i + j] += &factor.coefficients[i] * &self.coefficients[j];
                }
            }
            Polynomial::new(&coefficients)
        }
    }

    pub fn scale<S: IsSubFieldOf<F>>(&self, factor: &FieldElement<S>) -> Self {
        let scaled_coefficients = self
            .coefficients
            .iter()
            .zip(std::iter::successors(Some(FieldElement::one()), |x| {
                Some(x * factor)
            }))
            .map(|(coeff, power)| power * coeff)
            .collect();
        Self {
            coefficients: scaled_coefficients,
        }
    }

    pub fn scale_coeffs(&self, factor: &FieldElement<F>) -> Self {
        let scaled_coefficients = self
            .coefficients
            .iter()
            .map(|coeff| factor * coeff)
            .collect();
        Self {
            coefficients: scaled_coefficients,
        }
    }

    /// Returns a vector of polynomials [p₀, p₁, ..., p_{d-1}], where d is `number_of_parts`, such that `self` equals
    /// p₀(Xᵈ) + Xp₁(Xᵈ) + ... + X^(d-1)p_{d-1}(Xᵈ).
    ///
    /// Example: if d = 2 and `self` is 3 X^3 + X^2 + 2X + 1, then `poly.break_in_parts(2)`
    /// returns a vector with two polynomials `(p₀, p₁)`, where p₀ = X + 1 and p₁ = 3X + 2.
    pub fn break_in_parts(&self, number_of_parts: usize) -> Vec<Self> {
        let coef = self.coefficients();
        let mut parts: Vec<Self> = Vec::with_capacity(number_of_parts);
        for i in 0..number_of_parts {
            let coeffs: Vec<_> = coef
                .iter()
                .skip(i)
                .step_by(number_of_parts)
                .cloned()
                .collect();
            parts.push(Polynomial::new(&coeffs));
        }
        parts
    }

    pub fn to_extension<L: IsField>(self) -> Polynomial<FieldElement<L>>
    where
        F: IsSubFieldOf<L>,
    {
        Polynomial {
            coefficients: self
                .coefficients
                .into_iter()
                .map(|x| x.to_extension::<L>())
                .collect(),
        }
    }
}

pub fn pad_with_zero_coefficients_to_length<F: IsField>(
    pa: &mut Polynomial<FieldElement<F>>,
    n: usize,
) {
    pa.coefficients.resize(n, FieldElement::zero());
}

/// Pads polynomial representations with minimum number of zeros to match lengths.
pub fn pad_with_zero_coefficients<L: IsField, F: IsSubFieldOf<L>>(
    pa: &Polynomial<FieldElement<F>>,
    pb: &Polynomial<FieldElement<L>>,
) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<L>>) {
    let mut pa = pa.clone();
    let mut pb = pb.clone();

    if pa.coefficients.len() > pb.coefficients.len() {
        pad_with_zero_coefficients_to_length(&mut pb, pa.coefficients.len());
    } else {
        pad_with_zero_coefficients_to_length(&mut pa, pb.coefficients.len());
    }
    (pa, pb)
}

impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, a_polynomial: &Polynomial<FieldElement<L>>) -> Self::Output {
        let (pa, pb) = pad_with_zero_coefficients(self, a_polynomial);
        let iter_coeff_pa = pa.coefficients.iter();
        let iter_coeff_pb = pb.coefficients.iter();
        let new_coefficients = iter_coeff_pa.zip(iter_coeff_pb).map(|(x, y)| x + y);
        let new_coefficients_vec = new_coefficients.collect::<Vec<FieldElement<L>>>();
        Polynomial::new(&new_coefficients_vec)
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, a_polynomial: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + &a_polynomial
    }
}

impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, a_polynomial: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + a_polynomial
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, a_polynomial: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self + &a_polynomial
    }
}

impl<F: IsField> ops::Neg for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn neg(self) -> Polynomial<FieldElement<F>> {
        let neg = self
            .coefficients
            .iter()
            .map(|x| -x)
            .collect::<Vec<FieldElement<F>>>();
        Polynomial::new(&neg)
    }
}

impl<F: IsField> ops::Neg for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn neg(self) -> Polynomial<FieldElement<F>> {
        -&self
    }
}

impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, substrahend: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self + (-substrahend)
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, substrahend: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - &substrahend
    }
}

impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, substrahend: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - substrahend
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, substrahend: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self - &substrahend
    }
}

impl<F> ops::Div<Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>>
where
    F: IsField,
{
    type Output = Polynomial<FieldElement<F>>;

    fn div(self, dividend: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self.div_with_ref(&dividend)
    }
}

impl<F: IsField> ops::Mul<&Polynomial<FieldElement<F>>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self.mul_with_ref(factor)
    }
}

impl<F: IsField> ops::Mul<Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self * &factor
    }
}

impl<F: IsField> ops::Mul<Polynomial<FieldElement<F>>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self * &factor
    }
}

impl<F: IsField> ops::Mul<&Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self * factor
    }
}

/* Operations between Polynomials and field elements */
/* Multiplication field element at left */
impl<F, L> ops::Mul<FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        let new_coefficients = self
            .coefficients
            .iter()
            .map(|value| &multiplicand * value)
            .collect();
        Polynomial {
            coefficients: new_coefficients,
        }
    }
}

impl<F, L> ops::Mul<&FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        self.clone() * multiplicand.clone()
    }
}

impl<F, L> ops::Mul<FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        self * &multiplicand
    }
}

impl<F, L> ops::Mul<&FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self * multiplicand
    }
}

/* Multiplication field element at right */
impl<F, L> ops::Mul<&Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        multiplicand * self
    }
}

impl<F, L> ops::Mul<Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &multiplicand * self
    }
}

impl<F, L> ops::Mul<&Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        multiplicand * self
    }
}

impl<F, L> ops::Mul<Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &multiplicand * &self
    }
}

/* Addition field element at left */
impl<F, L> ops::Add<&FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        Polynomial::new_monomial(other.clone(), 0) + self
    }
}

impl<F, L> ops::Add<FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self + &other
    }
}

impl<F, L> ops::Add<FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        self + &other
    }
}

impl<F, L> ops::Add<&FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self + other
    }
}

/* Addition field element at right */
impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        Polynomial::new_monomial(self.clone(), 0) + other
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + &other
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self + &other
    }
}

impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + other
    }
}

/* Substraction field element at left */
impl<F, L> ops::Sub<&FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        -Polynomial::new_monomial(other.clone(), 0) + self
    }
}

impl<F, L> ops::Sub<FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self - &other
    }
}

impl<F, L> ops::Sub<FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        self - &other
    }
}

impl<F, L> ops::Sub<&FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self - other
    }
}

/* Substraction field element at right */
impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        Polynomial::new_monomial(self.clone(), 0) - other
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - &other
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self - &other
    }
}

impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - other
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum InterpolateError {
    UnequalLengths(usize, usize),
    NonUniqueXs,
}

impl Display for InterpolateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpolateError::UnequalLengths(x, y) => {
                write!(f, "xs and ys must be the same length. Got: {x} != {y}")
            }
            InterpolateError::NonUniqueXs => write!(f, "xs values should be unique."),
        }
    }
}
