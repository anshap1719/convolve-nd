use ndarray::Array1;
use convolve_nd::Convolution;
use convolve_nd::kernel::SeparableKernel;

fn main() {
    let mut data = Array1::from(vec![0.12312, 0.43654, 0.3466764, 0.78775, 0.198, 0.823478, 0.961253, 0.15647, 0.49801, 0.12682]);
    data.convolve(SeparableKernel::new([1. / 4., 1. / 2., 1. / 4.]), 1);
    
    assert_eq!(data, Array1::from(vec![0.27983, 0.3748966, 0.46399987, 0.5593749, 0.44471323, 0.7632305, 0.7105516, 0.38037542, 0.37580386, 0.18906596]));
}