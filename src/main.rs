use byteorder::{LittleEndian, ReadBytesExt};
use glam::{Quat, Vec2, Vec3};

use std::io::Cursor;
use std::net::UdpSocket;

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

    loop {
        // Receive data
        let (number_of_bytes, _src_addr) = socket.recv_from(&mut buf).unwrap();

        let packet_data = &buf[..number_of_bytes]; // You should fill this with actual UDP packet data
        let mut parsed_data = ParsedData::default();
        parsed_data.read_from_packet(&packet_data);
        println!("{:?}", parsed_data);
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
