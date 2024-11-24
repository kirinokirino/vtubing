#![allow(unused)]

use anyhow::{Error, Result};
use glam::Vec2;
use itertools::Itertools;
use ndarray::{s, Array};
use ort::{GraphOptimizationLevel, Session};
extern crate opencv;
use opencv::{
    core::{Size2i, Vec3b, Vec3f, VecN, CV_32FC1, CV_32FC3},
    imgproc,
    prelude::*,
    videoio::{
        VideoCapture, VideoCaptureTrait, CAP_ANY, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
    },
};

use std::f32::consts::PI;
use std::io::Write;

mod face_info;
mod feature;
mod feature_extractor;
mod math;
mod remedian;
mod tracker;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

pub fn main() {
    let path = "/home/k/Documents/Rust/k/vtubing/onnx/lib/libonnxruntime.so.1.20.0";
    ort::init_from(path).commit().unwrap();

    let num_threads = 4;
    let face_detector = FaceDetector::new(num_threads, 0.6);

    // Open the video capture (0 for default webcam, or a video file path)
    let mut capture = VideoCapture::new(0, CAP_ANY).unwrap(); // 0 for default webcam
                                                              // Set the capture resolution to 640x480
    capture.set(CAP_PROP_FRAME_WIDTH, 640.0).unwrap();
    capture.set(CAP_PROP_FRAME_HEIGHT, 480.0).unwrap();
    if !capture.is_opened().unwrap() {
        panic!("Error: Couldn't open video capture.");
    }

    let mut preprocessor = ImagePreprocessor::new();
    loop {
        // Capture a frame from the video capture
        capture.read(&mut preprocessor.frame).unwrap();
        if preprocessor.frame.empty() {
            eprintln!("Error: No frame captured.");
            continue;
        }

        let input = preprocessor.preprocess();

        match face_detector.detect_face(input) {
            Ok(face) => panic!("{:?}", face),
            Err(err) => eprintln!("No face detected, {:?}", err),
        }

        if face_detector.should_stop() {
            break;
        }

        // println!("{:?}", input.shape());
        // println!("{:?}", input.slice(s![0, .., .., ..]));

        // results = []
        // for det in detections[0:1]:

        //     y, x = det // 56, det % 56
        //     c = outputs[0, 0, y, x]
        //     r = outputs[0, 1, y, x] * 112.
        //     x *= 4
        //     y *= 4
        //     r *= 1.0
        //     if c < self.threshold:
        //         break
        //     results.append((x - r, y - r, 2 * r, 2 * r * 1.0))
        // results = np.array(results).astype(np.float32)
        // if results.shape[0] > 0:
        //     results[:, [0,2]] *= frame.shape[1] / 224.
        //     results[:, [1,3]] *= frame.shape[0] / 224.
        // return results

        //println!("Frame captured: {:?}", frame);
    }

    // Close the video capture and window
    capture.release().unwrap();
}

struct FaceDetector {
    detection: Session,
    threshold: f32,
}

impl FaceDetector {
    fn new(num_threads: usize, threshold: f32) -> Self {
        let detection = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(num_threads)
            .unwrap()
            .commit_from_file("models/mnv3_detection_opt.onnx")
            .unwrap();

        Self {
            detection,
            threshold,
        }
    }

    fn detect_face(&self, input: ndarray::Array4<f32>) -> Result<[f32; 4]> {
        // let mut canvas = Canvas::new();

        // canvas.pen_color = [255, 0, 0, 255];
        // let pic = input.slice(s![0usize, .., .., ..]);
        // let (_, w, h) = pic.dim();
        // for x in 0..w {
        //     for y in 0..h {
        //         let r = pic[[0, y, x]] * 255.0;
        //         let g = pic[[1, y, x]] * 255.0;
        //         let b = pic[[2, y, x]] * 255.0;
        //         canvas.pen_color = [r as u8, g as u8, b as u8, 255];
        //         canvas.draw_point(Vec2::new(x as f32, y as f32));
        //     }
        // }
        //canvas.display();
        // return Err(Error::msg("test"));

        let output = self
            .detection
            .run(ort::inputs!("input" => input).unwrap())
            .unwrap();

        let mut predictions = output["output"]
            .try_extract_tensor::<f32>()
            .unwrap()
            .to_owned();
        let maxpool = output["maxpool"].try_extract_tensor::<f32>().unwrap();

        // outputs[0, 0, outputs[0, 0] != maxpool[0, 0]] = 0
        let pred_data = predictions.slice_mut(s![0, 0, .., ..]);
        let maxpool_data = maxpool.slice(s![0, 0, .., ..]);
        // for (pred, &max) in pred_data.iter_mut().zip(maxpool_data.iter()) {
        //     if *pred != max {
        //         *pred = 0.0;
        //     }
        // }

        // detections = np.flip(np.argsort(outputs[0,0].flatten()))
        let detections: Vec<_> = Array::from_iter(pred_data.iter().cloned())
            .into_iter()
            .enumerate()
            .sorted_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap())
            .map(|(i, _)| i)
            .collect();

        let test = detections.iter().take_while(|el| el > &&10).collect_vec();
        let canvas = Canvas::new();

        // canvas.pen_color = [255, 0, 0, 255];
        // for i in test {
        //     let x = i % 56;
        //     let y = i / 56;

        //     canvas.draw_circle(
        //         Vec2::new(
        //             map(x as f32, 0.0, 56.0, 0.0, 640.0),
        //             map(y as f32, 0.0, 56.0, 0.0, 480.0),
        //         ),
        //         10.0,
        //     );
        // }
        // canvas.display();

        let detection = detections[0];

        let y = detection / 56;
        if y >= 56 {
            eprintln!("y {} >= 56", detection);
            return Err(Error::msg("y >= 56"));
        }
        let x = detection % 56;
        let c = predictions[[0, 0, y, x]];
        let r = predictions[[0, 1, y, x]] * 112.0;
        let x = x * 4;
        let y = y * 4;

        // for i in 0..56 {
        //     for j in 0..56 {
        //         let color = predictions.slice(s![0, 1, i, j]).into_scalar();
        //         let scaled = map(*color, 0.0, 0.01, 0.0, 255.0);
        //         canvas.pen_color = [scaled as u8, scaled as u8, scaled as u8, 255];
        //         canvas.draw_circle(Vec2::new(
        //             map(i as f32, 0.0, 56.0, 0.0, 640.0),
        //             map(j as f32, 0.0, 56.0, 0.0, 480.0),
        //         ),
        //             7.0,
        //         );
        //     }
        // }

        // canvas.pen_color = [255, 0, 0, 255];
        // let v = Vec2::new(
        //     map(x as f32 - r, -r, 56.0 * 4.0, 0.0, 640.0),
        //     map(y as f32 - r, -r, 56.0 * 4.0, 0.0, 480.0),
        // );
        // canvas.draw_square(v, Vec2::new(r * 2.0, r * 2.0) + v);
        // println!("{:?}", v);
        // println!("{:?}", r);
        // canvas.display();
        if c < self.threshold {
            return Err(Error::msg(format!("confidence {c} < {}", self.threshold)));
        }

        let result = [x as f32 - r, y as f32 - r, 2.0 * r, 2.0 * r * 1.0];

        println!("{:?}", result);
        Ok(result)
    }

    fn should_stop(&self) -> bool {
        false
    }
}

struct ImagePreprocessor {
    std_224: [f32; 3],
    mean_224: [f32; 3],
    pub frame: Mat,
    resized: Mat,
    rgb: Mat,
    normalized: Mat,
}

impl ImagePreprocessor {
    fn new() -> Self {
        let std_224 = [0.017125, 0.017507, 0.017429];
        let mean_224 = [-2.11790, -2.03571, -1.80444];

        let frame = Mat::default();
        let resized = Mat::default();
        let rgb = Mat::default();
        let normalized = Mat::default();
        Self {
            std_224,
            mean_224,
            frame,
            resized,
            rgb,
            normalized,
        }
    }

    fn preprocess(&mut self) -> ndarray::Array4<f32> {
        imgproc::resize(
            &self.frame,
            &mut self.resized,
            Size2i::new(224, 224),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )
        .unwrap();

        // Step 2: Convert from BGR to RGB by reversing channels
        imgproc::cvt_color(&self.resized, &mut self.rgb, imgproc::COLOR_BGR2RGB, 0).unwrap();

        // // Step 3: Normalize the image by applying the mean and std deviation
        // // We need to cast the Mat to a type that can hold the resulting floating point values (CV_32F)
        self.rgb
            .convert_to(&mut self.normalized, CV_32FC1, 1.0, 0.0)
            .unwrap();

        // Normalize using (pixel - mean) / std
        for row in 0..self.normalized.rows() {
            for col in 0..self.normalized.cols() {
                let pixel: &mut VecN<f32, 3> = self.normalized.at_2d_mut(row, col).unwrap();
                for channel in 0..3usize {
                    // pixel[channel] =
                    //     (pixel[channel] - self.mean_224[channel]) / self.std_224[channel];

                    pixel[channel] *= self.std_224[channel]; // + self.mean_224[channel];
                }
            }
        }

        // let mut canvas = Canvas::new();
        // canvas.pen_color = [255, 0, 0, 255];
        // println!("{:?}", self.normalized.dims());
        // for y in 0..self.normalized.rows() {
        //     for x in 0..self.normalized.cols() {
        //         let pixel = self.normalized.at_2d::<Vec3f>(y, x).unwrap();
        //         canvas.pen_color = [
        //             map(pixel[0], 0.0, 1.0, 0.0, 255.0) as u8,
        //             map(pixel[1], 0.0, 1.0, 0.0, 255.0) as u8,
        //             map(pixel[2], 0.0, 1.0, 0.0, 255.0) as u8,
        //             255,
        //         ];
        //         canvas.draw_point(Vec2::new(x as f32, y as f32));
        //     }
        // }
        // canvas.display();

        // Convert `Mat` to `ndarray::Array4`
        let array = ImagePreprocessor::mat_to_ndarray(&self.normalized).unwrap();

        // Step 1: Expand dimensions (add a batch dimension)
        // Adds a batch dimension at axis 0

        //let transposed = expanded.permuted_axes([0, 1, 2, 3]);
        array.insert_axis(ndarray::Axis(0))
    }

    // Convert a Mat (OpenCV) to ndarray (Rust)
    fn mat_to_ndarray(mat: &Mat) -> opencv::Result<ndarray::Array3<f32>> {
        if mat.typ() != CV_32FC3 {
            panic!("Expected CV_32FC3, got {:?}", mat.typ());
        }
        // Get the number of rows, columns, and channels
        let rows = mat.rows();
        let cols = mat.cols();
        let channels = mat.channels();

        let mat_data = mat.data_typed::<Vec3f>().unwrap();
        let flattened_data: Vec<f32> = mat_data
            .iter()
            .flat_map(|vec| vec.0) // `vec.0` accesses the internal `[f32; 3]` in Vec3f
            .collect();

        let array = ndarray::Array::from_shape_vec(
            (channels as usize, rows as usize, cols as usize),
            flattened_data,
        );

        match array {
            Ok(array) => Ok(array),
            Err(err) => {
                panic!("{err}");
                // Err(opencv::Error {
                // code: opencv::core::StsError,
                // message: format!("ShapeError: {err}"),
            }
        }
    }

    fn debug_display_frame(&self) {
        let mut canvas = Canvas::new();
        canvas.pen_color = [255, 0, 0, 255];
        println!("{:?}", self.frame.dims());
        for y in 0..self.frame.rows() {
            for x in 0..self.frame.cols() {
                canvas.pen_color = [
                    self.frame.at_2d::<Vec3b>(y, x).unwrap()[2],
                    self.frame.at_2d::<Vec3b>(y, x).unwrap()[1],
                    self.frame.at_2d::<Vec3b>(y, x).unwrap()[0],
                    255,
                ];
                canvas.draw_point(Vec2::new(x as f32, y as f32));
            }
        }
        canvas.display();
    }
}

struct Canvas {
    pub buffer: Vec<u8>,
    pub pen_color: [u8; 4],
}

impl Canvas {
    pub fn new() -> Self {
        let buffer = vec![255u8; WIDTH * HEIGHT * 4];
        let pen_color = [255, 255, 255, 255];
        Self { buffer, pen_color }
    }

    pub fn transparent(&mut self) {
        self.pen_color = [0, 0, 0, 0];
    }

    pub fn display(&self) {
        let file = std::fs::File::options()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open("/tmp/imagesink")
            .unwrap();
        let size = 640 * 480 * 4;
        file.set_len(size.try_into().unwrap()).unwrap();
        let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file).unwrap() };
        if let Some(err) = mmap.lock().err() {
            panic!("{err}");
        }
        let _ = (&mut mmap[..]).write_all(self.buffer.as_slice());
    }

    fn draw_curve(&mut self, start: Vec2, control: Vec2, end: Vec2) {
        let points = start.distance(control) + control.distance(end) + end.distance(start);
        for i in 1..points as usize {
            let proportion = i as f32 / points;
            let path1 = control - start;
            let point1 = start + path1 * proportion;
            let path2 = end - control;
            let point2 = control + path2 * proportion;
            let path3 = point2 - point1;
            let point3 = point1 + path3 * proportion;
            self.draw_point(point3);
        }
    }

    fn draw_line(&mut self, from: Vec2, to: Vec2) {
        let delta = to - from;
        let normalized = delta.normalize();
        for step in 0..delta.length() as usize {
            let magnitude = step as f32;
            let x = from.x + normalized.x * magnitude;
            let y = from.y + normalized.y * magnitude;
            self.draw_point(Vec2::new(x, y));
        }
    }

    fn draw_circle(&mut self, pos: Vec2, radius: f32) {
        let left_x = (pos.x - radius) as usize;
        let right_x = (pos.x + radius) as usize;
        let top_y = (pos.y - radius) as usize;
        let bottom_y = (pos.y + radius) as usize;
        for offset_x in left_x..=right_x {
            for offset_y in top_y..=bottom_y {
                if ((offset_x as f32 - pos.x).powi(2) + (offset_y as f32 - pos.y).powi(2)).sqrt()
                    < radius
                {
                    self.draw_point(Vec2::new(offset_x as f32, offset_y as f32));
                }
            }
        }
    }

    fn draw_arc(&mut self, center: Vec2, radius: f32, angle_spread: f32, direction: f32) {
        let steps = (2.0 * radius * PI) as usize + 1;
        let start = direction - angle_spread / 2.0;
        let end = direction + angle_spread / 2.0;
        for step in 0..steps {
            let arc_point =
                Vec2::from_angle(map(step as f32, 0.0, steps as f32, start, end)) * radius + center;
            self.draw_point(arc_point);
        }
    }

    fn draw_square(&mut self, top_left: Vec2, bottom_right: Vec2) {
        for offset_x in top_left.x as usize..=bottom_right.x as usize {
            for offset_y in top_left.y as usize..=bottom_right.y as usize {
                self.draw_point(Vec2::new(offset_x as f32, offset_y as f32));
            }
        }
    }

    fn draw_point(&mut self, pos: Vec2) {
        if pos.x >= 640.0 || pos.x < 0.0 || pos.y >= 480.0 || pos.y < 0.0 {
            return;
        }
        let buffer_idx = self.idx(pos.x as usize, pos.y as usize);
        // if (buffer_idx + 3) > self.buffer.len() {
        //     // TODO err?
        //     return;
        // }
        self.point_blend(buffer_idx);
    }

    fn point_blend(&mut self, buffer_idx: usize) {
        let [r, g, b, a] = self.pen_color;

        if a == 0 {
            return;
        } else if a == 255 {
            self.point_replace(buffer_idx);
            return;
        }

        let mix = a as f32 / 255.0;
        let [dst_r, dst_g, dst_b, dst_a] = [
            self.buffer[buffer_idx] as f32,
            self.buffer[buffer_idx + 1] as f32,
            self.buffer[buffer_idx + 2] as f32,
            self.buffer[buffer_idx + 3] as f32,
        ];

        self.buffer[buffer_idx] = ((r as f32 * mix) + (dst_r * (1.0 - mix))) as u8;
        self.buffer[buffer_idx + 1] = ((g as f32 * mix) + (dst_g * (1.0 - mix))) as u8;
        self.buffer[buffer_idx + 2] = ((b as f32 * mix) + (dst_b * (1.0 - mix))) as u8;
        self.buffer[buffer_idx + 3] = ((a as f32 * mix) + (dst_a * (1.0 - mix))) as u8;
    }

    fn point_replace(&mut self, buffer_idx: usize) {
        self.buffer[buffer_idx] = self.pen_color[0];
        self.buffer[buffer_idx + 1] = self.pen_color[1];
        self.buffer[buffer_idx + 2] = self.pen_color[2];
        self.buffer[buffer_idx + 3] = self.pen_color[3];
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        (x + y * WIDTH) * 4
    }
}

pub fn map(value: f32, start1: f32, stop1: f32, start2: f32, stop2: f32) -> f32 {
    (value - start1) / (stop1 - start1) * (stop2 - start2) + start2
}
