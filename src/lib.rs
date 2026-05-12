//! Library facade for the hash-mining workspace.
//!
//! Exposes the OpenCL GPU keccak256 PoW backend (`gpu` module) so it can be
//! shared by all binaries in the package (`hash-miner-rs`, `pfft-miner-rs`).

#[cfg(feature = "gpu")]
pub mod gpu;
