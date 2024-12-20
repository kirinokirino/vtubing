#![allow(unused)]
use glam::Vec2;
use rscam::{Camera, Config, Frame};
use rusty_yunet::FaceLandmarks;

use std::f32::consts::PI;
use std::io::{Read, Write};
const WIDTH: usize = 640;
const HEIGHT: usize = 480;

fn yuyv_to_bgr(yuyv_data: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert!(
        yuyv_data.len() == width * height * 2,
        "Invalid YUYV data size"
    );

    let mut bgr_data = Vec::with_capacity(width * height * 3);

    for chunk in yuyv_data.chunks_exact(4) {
        let y0 = chunk[0] as f32;
        let u = chunk[1] as f32 - 128.0;
        let y1 = chunk[2] as f32;
        let v = chunk[3] as f32 - 128.0;

        // Convert YUV to BGR for the first pixel
        bgr_data.extend_from_slice(&yuv_to_bgr(y0, u, v));

        // Convert YUV to BGR for the second pixel
        bgr_data.extend_from_slice(&yuv_to_bgr(y1, u, v));
    }

    bgr_data
}

fn yuv_to_bgr(y: f32, u: f32, v: f32) -> [u8; 3] {
    let r = (y + 1.402 * v).clamp(0.0, 255.0) as u8;
    let g = (y - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
    let b = (y + 1.772 * u).clamp(0.0, 255.0) as u8;

    [b, g, r] // Swapped order to produce BGR
}
pub fn main() {
    let mut canvas = Canvas::new();

    let mut camera = Camera::new("/dev/video0").unwrap();

    // Configure the camera
    camera
        .start(&Config {
            interval: (1, 30), // 30 FPS
            resolution: (640, 480),
            // [0]: 'YUYV' (YUYV 4:2:2)
            // [1]: 'MJPG' (Motion-JPEG, compressed)
            // [2]: 'NV12' (Y/UV 4:2:0)
            format: b"YUYV", // Supported formats can be checked via `v4l2-ctl --list-formats`
            ..Default::default()
        })
        .unwrap();

    let mut memory = Vec::new();
    loop {
        canvas.pen_color = [0, 0, 0, 255];
        canvas.draw_square(Vec2::new(0.0, 0.0), Vec2::new(640.0, 480.0));
        let frame = camera.capture().unwrap();

        let mut image = yuyv_to_bgr(
            &frame,
            frame.resolution.0 as usize,
            frame.resolution.1 as usize,
        );

        let faces = rusty_yunet::detect_faces(
            &image,
            frame.resolution.0 as usize,
            frame.resolution.1 as usize,
        )
        .unwrap();

        for face in faces.into_iter().by_ref() {
            let c = (face.confidence() * 255.0) as u8;
            canvas.pen_color = [c, c, c, 255];

            let FaceLandmarks {
                right_eye,
                left_eye,
                nose,
                mouth_right,
                mouth_left,
            } = smoothed(face.landmarks(), &mut memory);
            let eyes_center = (right_eye + left_eye) / 2.0;
            let offset = eyes_center - nose;
            canvas.draw_line(right_eye + offset, nose + offset);
            canvas.draw_line(left_eye + offset, nose + offset);
            let size = Vec2::new(10.0, 10.0);
            canvas.draw_square(left_eye - size / 2.0, left_eye + size / 2.0);
            canvas.draw_square(right_eye - size / 2.0, right_eye + size / 2.0);
            canvas.draw_curve(mouth_left, nose, mouth_right);
            // canvas.draw_circle(*right_eye, 5.0);
            // canvas.draw_circle(*left_eye, 5.0);
            // canvas.draw_circle(*nose, 5.0);
            // canvas.draw_circle(*mouth_right, 5.0);
            // canvas.draw_circle(*mouth_left, 5.0);
        }
        canvas.display();
        std::thread::sleep(std::time::Duration::from_millis(1000 / 60));
    }
}

fn smoothed(next_landmarks: &FaceLandmarks, mut memory: &mut Vec<FaceLandmarks>) -> FaceLandmarks {
    if memory.is_empty() {
        memory.clear();
        memory.push(next_landmarks.clone());
        next_landmarks.clone()
    } else {
        let last_landmarks = &memory[0];
        let res = FaceLandmarks {
            right_eye: (last_landmarks.right_eye + next_landmarks.right_eye) / 2.0,
            left_eye: (last_landmarks.left_eye + next_landmarks.left_eye) / 2.0,
            nose: (last_landmarks.nose + next_landmarks.nose) / 2.0,
            mouth_left: (last_landmarks.mouth_left + next_landmarks.mouth_left) / 2.0,
            mouth_right: (last_landmarks.mouth_right + next_landmarks.mouth_right) / 2.0,
        };
        memory.clear();
        memory.push(next_landmarks.clone());
        res
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
            .truncate(false)
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
