// Library Modules
pub mod err;
pub mod layer;
pub mod network;
pub mod util;

// Main Types used in the crate
use ndarray::{Array, ArrayView, Ix1, Ix2};

// Setup basic types to be used in the crate
pub type AIVec = Array<f64, Ix1>;
pub type AIVecRef<'a> = ArrayView<'a, f64, Ix1>;
pub type AIWeights = Array<f64, Ix2>;