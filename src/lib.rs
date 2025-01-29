pub mod kernel;
pub mod convolve;
pub mod array;
pub(crate) mod dimensions;
pub mod rescale;
pub use convolve::Convolution;

pub mod exports {
    pub use ndarray;
}