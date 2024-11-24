use glam::{Mat2, Vec2};
use std::f32::consts::PI;

pub fn clamp_to_image(point: Vec2, width: i32, height: i32) -> (i32, i32) {
    let mut x = point.x;
    let mut y = point.y;

    if x < 0.0 {
        x = 0.0;
    }
    if y < 0.0 {
        y = 0.0;
    }
    if x >= width as f32 {
        x = (width - 1) as f32;
    }
    if y >= height as f32 {
        y = (height - 1) as f32;
    }

    (x as i32, (y + 1.0) as i32)
}

pub fn compensate_rotation(p1: Vec2, p2: Vec2) -> (Vec2, f32) {
    let a = angle(p1, p2);
    let rotated = rotate(p1, p2, a);
    (rotated, a)
}

pub fn rotate(origin: Vec2, point: Vec2, angle: f32) -> Vec2 {
    let angle = -angle;
    let offset = point - origin;

    let rot = Mat2::from_angle(angle);
    origin + rot.mul_vec2(offset)
}

pub fn angle(p1: Vec2, p2: Vec2) -> f32 {
    let diff = p2 - p1;
    let mut angle = diff.y.atan2(diff.x);
    if angle < 0.0 {
        angle += 2.0 * PI;
    }
    angle
}

pub fn compensate(p1: Vec2, p2: Vec2) -> (Vec2, f32) {
    let a = angle(p1, p2);
    (rotate(p1, p2, a), a)
}

pub fn logit(p: f32, factor: f32) -> f32 {
    let p = if p >= 1.0 {
        0.9999999
    } else if p <= 0.0 {
        0.0000001
    } else {
        p
    };
    let p = p / (1.0 - p);
    p.ln() / factor
}

pub fn matrix_to_quaternion(m: [[f32; 3]; 3]) -> [f32; 4] {
    let mut t = 0.0;
    let mut q = [0.0; 4];

    if m[2][2] < 0.0 {
        if m[0][0] > m[1][1] {
            t = 1.0 + m[0][0] - m[1][1] - m[2][2];
            q = [t, m[0][1] + m[1][0], m[2][0] + m[0][2], m[1][2] - m[2][1]];
        } else {
            t = 1.0 - m[0][0] + m[1][1] - m[2][2];
            q = [m[0][1] + m[1][0], t, m[1][2] + m[2][1], m[2][0] - m[0][2]];
        }
    } else if m[0][0] < -m[1][1] {
        t = 1.0 - m[0][0] - m[1][1] + m[2][2];
        q = [m[2][0] + m[0][2], m[1][2] + m[2][1], t, m[0][1] - m[1][0]];
    } else {
        t = 1.0 + m[0][0] + m[1][1] + m[2][2];
        q = [m[1][2] - m[2][1], m[2][0] - m[0][2], m[0][1] - m[1][0], t];
    }

    let scale = 0.5 / t.sqrt();
    [q[0] * scale, q[1] * scale, q[2] * scale, q[3] * scale]
}
