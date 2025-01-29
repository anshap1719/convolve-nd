use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Array3;
use convolve_nd::Convolution;
use convolve_nd::kernel::SeparableKernel;

fn main() {
    let img = image::open("sample.jpg").unwrap().to_rgb32f();
    let mut array = Array3::<f32>::zeros((img.height() as usize, img.width() as usize, 3));

    for (x, y, pixel) in img.enumerate_pixels() {
        let [r, g, b] = pixel.0;

        array[[y as usize, x as usize, 0]] = r;
        array[[y as usize, x as usize, 1]] = g;
        array[[y as usize, x as usize, 2]] = b;
    }

    let kernel = SeparableKernel::new([1. / 4., 1. / 2., 1. / 4.]);

    array.convolve(kernel, 4);

    let mut image_buffer = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());

    for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
        let r = array[[y as usize, x as usize, 0]];
        let g = array[[y as usize, x as usize, 1]];
        let b = array[[y as usize, x as usize, 2]];

        *pixel = Rgb([r, g, b]);
    }

    DynamicImage::ImageRgb32F(image_buffer).to_rgb8().save("output.jpg").unwrap();
}