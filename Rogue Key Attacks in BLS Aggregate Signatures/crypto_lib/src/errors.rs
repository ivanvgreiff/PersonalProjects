use std::io;

use math_lib::errors::DeserializationError;

#[derive(Debug)]
pub enum SrsFromFileError {
    FileError(io::Error),
    DeserializationError(math_lib::errors::DeserializationError),
}

impl From<math_lib::errors::DeserializationError> for SrsFromFileError {
    fn from(err: DeserializationError) -> SrsFromFileError {
        match err {
            DeserializationError::InvalidAmountOfBytes => {
                SrsFromFileError::DeserializationError(DeserializationError::InvalidAmountOfBytes)
            }

            DeserializationError::FieldFromBytesError => {
                SrsFromFileError::DeserializationError(DeserializationError::FieldFromBytesError)
            }

            DeserializationError::PointerSizeError => {
                SrsFromFileError::DeserializationError(DeserializationError::PointerSizeError)
            }

            DeserializationError::InvalidValue => {
                SrsFromFileError::DeserializationError(DeserializationError::InvalidValue)
            }
        }
    }
}

impl From<std::io::Error> for SrsFromFileError {
    fn from(err: std::io::Error) -> SrsFromFileError {
        SrsFromFileError::FileError(err)
    }
}
