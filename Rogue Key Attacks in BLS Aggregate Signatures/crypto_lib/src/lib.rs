#![allow(clippy::op_ref)]
#![cfg_attr(not(feature = "std"), no_std)]
#[macro_use]
extern crate alloc;

#[cfg(feature = "std")]
pub mod errors;
pub mod hash;
