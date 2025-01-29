use ndarray::Array1;
use convolve_nd::Convolution;
use convolve_nd::kernel::SeparableKernel;

fn main() {
    let mut reader = hound::WavReader::open("sample.wav").unwrap();
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let mut samples = Array1::from(samples);

    // Simple linear interpolation kernel
    let kernel = SeparableKernel::new([1. / 4., 1. / 2., 1. / 4.]);

    samples.convolve(kernel, 4096);

    let mut writer = hound::WavWriter::create("output.wav", spec).unwrap();
    for (_, item) in samples.iter().enumerate() {
        writer
            .write_sample((item * i16::MAX as f32) as i16)
            .unwrap();
    }
}