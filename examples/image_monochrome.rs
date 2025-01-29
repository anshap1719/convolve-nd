use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::{Array2};
use convolve_nd::Convolution;
use convolve_nd::kernel::SeparableKernel;

fn main() {
    let img = image::open("sample.jpg").unwrap().to_luma32f();
    let mut array = Array2::<f32>::zeros((img.height() as usize, img.width() as usize));

    for (x, y, pixel) in img.enumerate_pixels() {
        let [l] = pixel.0;

        array[[y as usize, x as usize]] = l;
    }

    let kernel = SeparableKernel::new([1. / 4., 1. / 2., 1. / 4.]);

    array.convolve(kernel, 4);

    let mut image_buffer = ImageBuffer::<Luma<u8>, Vec<u8>>::new(img.width(), img.height());

    for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
        *pixel = Luma([(array[[y as usize, x as usize]].clamp(0., 1.) * u8::MAX as f32) as u8]);
    }

    DynamicImage::ImageLuma8(image_buffer).save("output.jpg").unwrap();
}