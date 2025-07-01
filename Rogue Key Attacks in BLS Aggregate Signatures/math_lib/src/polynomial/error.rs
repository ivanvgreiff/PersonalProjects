use std::fmt::Display;

#[derive(Debug, PartialEq, Eq)]
pub enum MultilinearError {
    InvalidMergeLength,
    IncorrectNumberofEvaluationPoints(usize, usize),
    ChisAndEvalsLengthMismatch(usize, usize),
}

impl Display for MultilinearError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MultilinearError::InvalidMergeLength => write!(f, "Invalid Merge Length"),
            MultilinearError::IncorrectNumberofEvaluationPoints(x, y) => {
                write!(f, "points: {x}, vars: {y}")
            }
            MultilinearError::ChisAndEvalsLengthMismatch(x, y) => {
                write!(f, "chis: {x}, evals: {y}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MultilinearError {}
