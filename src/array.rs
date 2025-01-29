use ndarray::{Array1, Array2, Array3};
use crate::Convolution;
use crate::kernel::SeparableKernel;
use crate::dimensions::DimensionIterator;

impl Convolution for Array1<f32> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: SeparableKernel<KERNEL_SIZE>, stride: usize) {
        let linear_kernel = kernel.values();

        let sample_length = self.len();
        
        for index in 0..sample_length {
            let mut signal_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let signal_index = Self::compute_signal_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    index,
                    sample_length
                );

                signal_sum += self[signal_index as usize] * *value;
            }

            self[index] = signal_sum;
        }
    }
}

impl Convolution for Array2<f32> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: SeparableKernel<KERNEL_SIZE>, stride: usize) {
        let linear_kernel = kernel.values();

        let (height, width) = self.dim();
        let dimensions = self.raw_dim();

        for (y, x) in dimensions.into_iter() {
            let mut signal_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let signal_index = Self::compute_signal_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    x,
                    width
                );

                signal_sum += self[[y, signal_index as usize]] * *value;
            }

            self[[y, x]] = signal_sum;
        }

        for (y, x) in dimensions.into_iter() {
            let mut signal_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let signal_index = Self::compute_signal_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    y,
                    height
                );

                signal_sum += self[[signal_index as usize, x]] * *value;
            }

            self[[y, x]] = signal_sum;
        }
    }
}

impl Convolution for Array3<f32> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: SeparableKernel<KERNEL_SIZE>, stride: usize) {
        let linear_kernel = kernel.values();
        let (height, width, _) = self.dim();
        let dimensions = self.raw_dim();

        for (y, x, z) in dimensions.into_iter() {
            let mut signal_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let signal_index = Self::compute_signal_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    x,
                    width
                );

                signal_sum += self[[y, signal_index as usize, z]] * *value;
            }

            self[[y, x, z]] = signal_sum;
        }

        for (y, x, z) in dimensions.into_iter() {
            let mut signal_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let signal_index = Self::compute_signal_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    y,
                    height
                );

                signal_sum += self[[signal_index as usize, x, z]] * *value;
            }

            self[[y, x, z]] = signal_sum;
        }
    }
}

pub(crate) trait Aggregate {
    fn min(&self) -> f32;
    fn max(&self) -> f32;
}

macro_rules! impl_aggregate {
    ($ty:ty) => {
        impl Aggregate for $ty {
            fn min(&self) -> f32 {
                *self.iter()
                    .reduce(|current, previous| {
                        if current < previous {
                            current
                        } else {
                            previous
                        }
                    })
                    .unwrap()
            }

            fn max(&self) -> f32 {
                *self
                    .iter()
                    .reduce(|current, previous| {
                        if current > previous {
                            current
                        } else {
                            previous
                        }
                    })
                    .unwrap()
            }
        }
    };
}

impl_aggregate!(Array1<f32>);
impl_aggregate!(Array2<f32>);
impl_aggregate!(Array3<f32>);