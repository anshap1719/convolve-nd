use ndarray::{Array1, Array2, Array3};
use crate::array::Aggregate;

#[derive(Copy, Clone)]
pub enum RescaleRange {
    Custom(f32, f32),
    Max,
}

impl RescaleRange {
    fn min(self) -> f32 {
        match self {
            RescaleRange::Custom(min, _) => min,
            RescaleRange::Max => 0.
        }
    }

    fn max(self) -> f32 {
        match self {
            RescaleRange::Custom(_, max) => max,
            RescaleRange::Max => 1.
        }
    }
}

pub trait Rescale {
    fn min(&self) -> f32;
    fn max(&self) -> f32;
    fn rescale(&mut self, range: RescaleRange);
    fn rescale_value(min: f32, max: f32, value: f32, range: RescaleRange) -> f32 {
        let (new_min, new_max) = (range.min(), range.max());
        let new_range = new_max - new_min;

        new_min + ((value - min) * new_range / (max - min))
    }
}

macro_rules! impl_rescale {
    ($ty:ty) => {
        impl Rescale for $ty {
            fn min(&self) -> f32 {
                Aggregate::min(self)
            }

            fn max(&self) -> f32 {
                Aggregate::max(self)
            }

            fn rescale(&mut self, range: RescaleRange) {
                let min = Rescale::min(self);
                let max = Rescale::max(self);

                for item in self.iter_mut() {
                    *item = Self::rescale_value(min, max, *item, range);
                }
            }
        }
    };
}

impl_rescale!(Array1<f32>);
impl_rescale!(Array2<f32>);
impl_rescale!(Array3<f32>);