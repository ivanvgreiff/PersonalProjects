use core::slice;
use std::mem::MaybeUninit;

use math_lib::{
    field::{element::FieldElement, traits::IsField},
    polynomial::InterpolateError,
    polynomial::Polynomial,
};

/// TODO: Task 1
///
/// Estimate the polynomial `f:F -> F` that map `x` in `xs` to `y` in `ys`
/// using the lagrange interpolation method
///
/// # Arguments
///
/// * `xs` – A vector of `x` points
/// * `ys` – A vector of `f(x)`, each item is a map of index-corresponding `x` in `xs`
pub fn interpolate<F: IsField>(
    xs: &[FieldElement<F>],
    ys: &[FieldElement<F>],
) -> Result<Polynomial<FieldElement<F>>, InterpolateError> {
    let n: usize = xs.len();
    if n != ys.len() {
        return Err(InterpolateError::UnequalLengths(n, ys.len()));
    }

    let mut scales = ys.to_owned();

    for i in 0..n {
        for j in 0..i {
            let diff = &xs[i] - &xs[j];
            if diff.eq(&FieldElement::zero()) {
                return Err(InterpolateError::NonUniqueXs);
            }
            let prev = scales[i].clone();
            scales[i] = prev * diff.inv().unwrap();
            let prev = scales[j].clone();
            scales[j] = prev * -diff.inv().unwrap();
        }
    }

    let one = FieldElement::<F>::one();

    let lagrange_polynomial = (0..n)
        .map(|i| -> Polynomial<FieldElement<F>> {
            let mut p = Polynomial::new(&[one.clone()]);
            for (j, fe) in xs.iter().enumerate() {
                if i != j {
                    let lagrange = Polynomial::new(&[one.clone(), fe.clone()]);
                    p = p.clone() * lagrange;
                }
            }
            p.scale(&scales[i])
        })
        .fold(
            Polynomial::new(&[one.clone()]),
            |prev, next| -> Polynomial<FieldElement<F>> { prev + next },
        );

    Ok(lagrange_polynomial)
}
