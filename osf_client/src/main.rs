use byteorder::{LittleEndian, ReadBytesExt};
use glam::{Quat, Vec2, Vec3};

use std::f32::consts::PI;
use std::io::{Cursor, Write};
use std::net::UdpSocket;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

const N_POINTS: usize = 68;
const PACKET_FRAME_SIZE: usize = 8
    + 4
    + 2 * 4
    + 2 * 4
    + 1
    + 4
    + 3 * 4
    + 3 * 4
    + 4 * 4
    + 4 * 68
    + 4 * 2 * 68
    + 4 * 3 * 70
    + 4 * 14;

fn main() {
    let socket = UdpSocket::bind("127.0.0.1:8080").unwrap();
    println!("Listening on 127.0.0.1:8080");

    let mut buf = [0; PACKET_FRAME_SIZE]; // Buffer to hold incoming data

    let mut canvas = Canvas::new();
    loop {
        // Receive data
        let (number_of_bytes, _src_addr) = socket.recv_from(&mut buf).unwrap();

        let packet_data = &buf[..number_of_bytes]; // You should fill this with actual UDP packet data
        let mut parsed_data = ParsedData::default();
        parsed_data.read_from_packet(&packet_data);
        canvas.pen_color = [0, 0, 0, 255];
        canvas.draw_square(Vec2::new(0., 0.), Vec2::new(WIDTH as f32, HEIGHT as f32));
        canvas.pen_color = [255, 255, 255, 255];
        for point in parsed_data.points_3d {
            let w = 320.0;
            let h = 240.0;
            canvas.draw_point(Vec2::new((point.x + 1.0) * w, (point.y + 1.0) * h));
        }
        canvas.display();
    }
}

#[derive(Debug, Default)]
struct OpenSeeFeatures {
    eye_left: f32,
    eye_right: f32,
    eyebrow_steepness_left: f32,
    eyebrow_up_down_left: f32,
    eyebrow_quirk_left: f32,
    eyebrow_steepness_right: f32,
    eyebrow_up_down_right: f32,
    eyebrow_quirk_right: f32,
    mouth_corner_up_down_left: f32,
    mouth_corner_in_out_left: f32,
    mouth_corner_up_down_right: f32,
    mouth_corner_in_out_right: f32,
    mouth_open: f32,
    mouth_wide: f32,
}

#[derive(Debug, Default)]
struct ParsedData {
    time: f64,
    id: i32,
    camera_resolution: Vec2,
    right_eye_open: f32,
    left_eye_open: f32,
    got_3d_points: bool,
    fit_3d_error: f32,
    raw_quaternion: Quat,
    raw_euler: Vec3,
    rotation: Vec3,
    translation: Vec3,
    confidence: Vec<f32>,
    points: Vec<Vec2>,
    points_3d: Vec<Vec3>,
    right_gaze: Quat,
    left_gaze: Quat,
    features: OpenSeeFeatures,
}

impl ParsedData {
    fn read_from_packet(&mut self, data: &[u8]) {
        let mut cursor = Cursor::new(data);
        self.time = cursor.read_f64::<LittleEndian>().unwrap();

        self.id = cursor.read_i32::<LittleEndian>().unwrap();

        self.camera_resolution = self.read_vector2(&mut cursor);
        self.right_eye_open = self.read_float(&mut cursor);
        self.left_eye_open = self.read_float(&mut cursor);

        let got_3d = cursor.read_u8().unwrap();
        self.got_3d_points = got_3d != 0;

        self.fit_3d_error = self.read_float(&mut cursor);
        self.raw_quaternion = self.read_quaternion(&mut cursor);

        // Convert quaternion as per the logic in C#
        self.raw_quaternion = Quat::from_xyzw(
            -self.raw_quaternion.x,
            self.raw_quaternion.y,
            -self.raw_quaternion.z,
            self.raw_quaternion.w,
        );

        self.raw_euler = self.read_vector3(&mut cursor);
        self.rotation = self.raw_euler.clone();
        self.rotation.z = (self.rotation.z - 90.0) % 360.0;
        self.rotation.x = -(self.rotation.x + 180.0) % 360.0;

        self.translation = Vec3::new(
            -self.read_float(&mut cursor),
            self.read_float(&mut cursor),
            -self.read_float(&mut cursor),
        );

        self.confidence = (0..N_POINTS)
            .map(|_| self.read_float(&mut cursor))
            .collect();
        self.points = (0..N_POINTS)
            .map(|_| self.read_vector2(&mut cursor))
            .collect();
        self.points_3d = (0..N_POINTS + 2)
            .map(|_| self.read_vector3(&mut cursor))
            .collect();

        // Calculate gaze (right_gaze and left_gaze)
        let (left_gaze, right_gaze) = ParsedData::calculate_gaze(&self.points_3d);
        self.right_gaze = left_gaze;
        self.left_gaze = right_gaze;

        // Read features
        self.features.eye_left = self.read_float(&mut cursor);
        self.features.eye_right = self.read_float(&mut cursor);
        self.features.eyebrow_steepness_left = self.read_float(&mut cursor);
        self.features.eyebrow_up_down_left = self.read_float(&mut cursor);
        self.features.eyebrow_quirk_left = self.read_float(&mut cursor);
        self.features.eyebrow_steepness_right = self.read_float(&mut cursor);
        self.features.eyebrow_up_down_right = self.read_float(&mut cursor);
        self.features.eyebrow_quirk_right = self.read_float(&mut cursor);
        self.features.mouth_corner_up_down_left = self.read_float(&mut cursor);
        self.features.mouth_corner_in_out_left = self.read_float(&mut cursor);
        self.features.mouth_corner_up_down_right = self.read_float(&mut cursor);
        self.features.mouth_corner_in_out_right = self.read_float(&mut cursor);
        self.features.mouth_open = self.read_float(&mut cursor);
        self.features.mouth_wide = self.read_float(&mut cursor);
    }

    fn read_vector2(&self, cursor: &mut Cursor<&[u8]>) -> Vec2 {
        Vec2::new(self.read_float(cursor), self.read_float(cursor))
    }

    fn read_vector3(&self, cursor: &mut Cursor<&[u8]>) -> Vec3 {
        Vec3::new(
            self.read_float(cursor),
            self.read_float(cursor),
            self.read_float(cursor),
        )
    }

    fn read_float(&self, cursor: &mut Cursor<&[u8]>) -> f32 {
        cursor.read_f32::<LittleEndian>().unwrap()
    }

    fn read_quaternion(&self, cursor: &mut Cursor<&[u8]>) -> Quat {
        Quat::from_xyzw(
            self.read_float(cursor),
            self.read_float(cursor),
            self.read_float(cursor),
            self.read_float(cursor),
        )
    }

    fn calculate_gaze(points_3d: &[Vec3]) -> (Quat, Quat) {
        // Right gaze calculation (points 66 and 68)
        let right_direction = swap_x(points_3d[66]) - swap_x(points_3d[68]);
        let right_gaze = Quat::from_rotation_arc(Vec3::Z, right_direction.normalize()) // LookRotation equivalent
            * Quat::from_axis_angle(Vec3::X, std::f32::consts::PI)   // 180 degrees around X axis
            * Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI); // 180 degrees around Y axis

        // Left gaze calculation (points 67 and 69)
        let left_direction = swap_x(points_3d[67]) - swap_x(points_3d[69]);
        let left_gaze = Quat::from_rotation_arc(Vec3::Z, left_direction.normalize()) // LookRotation equivalent
            * Quat::from_axis_angle(Vec3::X, std::f32::consts::PI)   // 180 degrees around X axis
            * Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI); // 180 degrees around Y axis

        (right_gaze, left_gaze)
    }
}

fn swap_x(v: Vec3) -> Vec3 {
    Vec3::new(-v.x, v.y, v.z)
}

struct Canvas {
    pub buffer: Vec<u8>,
    pub pen_color: [u8; 4],
}

impl Canvas {
    pub fn new() -> Self {
        let mut buffer = vec![255u8; WIDTH * HEIGHT * 4];
        let pen_color = [255, 255, 255, 255];
        Self { buffer, pen_color }
    }

    pub fn transparent(&mut self) {
        self.pen_color = [0, 0, 0, 0];
    }

    pub fn display(&self) {
        let file = std::fs::File::options()
            .create(true)
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
        let _ = (&mut mmap[..]).write_all(&self.buffer.as_slice());
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
                if ((offset_x as f32 - pos.x as f32).powi(2)
                    + (offset_y as f32 - pos.y as f32).powi(2))
                .sqrt()
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
