use glam::Vec2;
use ndarray::{s, Array1, Array2, Axis};

use std::collections::HashMap;
use std::ops::{DivAssign, SubAssign};

use crate::feature_extractor::FeatureExtractor;
use crate::math::angle;
use crate::tracker::Tracker;

pub struct FaceInfo {
    // Basic info
    pub id: i32,
    pub frame_count: i32,
    pub contour_pts: Vec<usize>,
    pub face_3d: Array2<f32>, // Shape: (3, N)

    // State flags and basic data
    pub alive: bool,
    pub conf: Option<f32>,
    pub coord: Option<Array1<f32>>, // Shape: (2,) for x,y coordinates

    // Landmarks and tracking
    pub lms: Option<Array2<f32>>, // Shape: (66, 3) - x,y coordinates and confidence
    pub eye_state: Option<Array2<f32>>, // Shape: (2, 4) - open, y, x, conf for each eye
    pub eye_blink: Vec<f32>,
    pub bbox: Option<(f32, f32, f32, f32)>, // y1, x1, height, width

    // 3D pose estimation
    pub rotation: Option<Array1<f32>>,    // Shape: (3,)
    pub translation: Option<Array1<f32>>, // Shape: (3,)
    pub quaternion: Option<Array1<f32>>,  // Shape: (4,)
    pub euler: Option<Array1<f32>>,       // Shape: (3,)
    pub success: Option<bool>,
    pub pnp_error: f32,
    pub pts_3d: Option<Array2<f32>>, // Shape: (70, 3)

    // Feature tracking
    pub features: FeatureExtractor,
    pub current_features: HashMap<String, f32>,

    // Contour and model adjustment
    pub contour: Array2<f32>,       // Shape: (21, 3)
    pub update_counts: Array2<f32>, // Shape: (66, 2)
    pub fail_count: i32,

    // Configuration
    pub limit_3d_adjustment: bool,
    pub update_count_delta: f32,
    pub update_count_max: f32,
    pub base_scale_v: Array1<f32>, // Shape: (3,)
    pub base_scale_h: Array1<f32>, // Shape: (3,)
}

impl FaceInfo {
    pub fn new(id: i32, tracker: &Tracker) -> Self {
        let mut face_info = Self {
            id,
            frame_count: -1,
            contour_pts: vec![0, 1, 8, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35],
            face_3d: tracker.face_3d.clone(),

            alive: false,
            conf: None,
            coord: None,

            lms: None,
            eye_state: None,
            eye_blink: vec![1.0, 1.0],
            bbox: None,

            rotation: None,
            translation: None,
            quaternion: None,
            euler: None,
            success: None,
            pnp_error: 0.0,
            pts_3d: None,

            features: if tracker.max_feature_updates < 1 {
                FeatureExtractor::new(0)
            } else {
                FeatureExtractor::new(tracker.max_feature_updates)
            },
            current_features: HashMap::new(),

            contour: Array2::zeros((21, 3)),
            update_counts: Array2::zeros((66, 2)),
            fail_count: 0,

            limit_3d_adjustment: true,
            update_count_delta: 75.0,
            update_count_max: 7500.0,
            // Extract vertical scale from face_3d
            base_scale_v: {
                let mut face_3d = tracker.face_3d.clone();
                let mut base_scale_v = face_3d.slice_mut(s![27..30, 1]);
                base_scale_v
                    .zip_mut_with(&tracker.face_3d.slice(s![28..31, 1]), |a, b| *a = *a - *b);
                base_scale_v.to_owned()
            },
            // Extract horizontal scale from face_3d
            base_scale_h: {
                let indices = [0, 36, 42];
                let indices2 = [16, 39, 45];
                tracker
                    .face_3d
                    .select(Axis(0), &indices)
                    .slice(s![.., 0])
                    .to_owned()
                    - tracker
                        .face_3d
                        .select(Axis(0), &indices2)
                        .slice(s![.., 0])
                        .to_owned()
            },
        };

        face_info.reset(tracker.max_feature_updates);
        face_info
    }

    pub fn reset(&mut self, max_feature_updates: i32) {
        self.alive = false;
        self.conf = None;
        self.lms = None;
        self.eye_state = None;
        self.rotation = None;
        self.translation = None;
        self.success = None;
        self.quaternion = None;
        self.euler = None;
        self.pnp_error = 0.0;
        self.pts_3d = None;
        self.eye_blink = vec![1.0, 1.0];
        self.bbox = None;

        if max_feature_updates < 1 {
            self.features = FeatureExtractor::new(0);
        }

        self.current_features.clear();
        self.contour = Array2::zeros((21, 3));
        self.update_counts = Array2::zeros((66, 2));
        self.update_contour();
        self.fail_count = 0;
    }

    pub fn update_contour(&mut self) {
        // Select rows based on contour_pts and all columns
        self.contour = self
            .face_3d
            .select(Axis(0), self.contour_pts.as_slice())
            .slice(s![.., ..3])
            .to_owned();
    }

    pub fn normalize_pts3d(&self, mut pts_3d: Array2<f32>) -> Array2<f32> {
        use ndarray::s;

        // Calculate angle using nose
        let nose_point = pts_3d.slice(s![30, ..2]).to_owned();
        pts_3d.slice_mut(s![..2, ..]).sub_assign(&nose_point);

        let alpha = angle(
            Vec2::new(pts_3d[[30, 0]], pts_3d[[30, 1]]),
            Vec2::new(pts_3d[[27, 0]], pts_3d[[27, 1]]),
        ) - std::f32::consts::FRAC_PI_2;

        // Create rotation matrix
        let rotation = Array2::from_shape_vec(
            (2, 2),
            vec![alpha.cos(), -alpha.sin(), alpha.sin(), alpha.cos()],
        )
        .unwrap();

        // Apply rotation
        let centered = &pts_3d - &pts_3d.slice(s![30, ..pts_3d.ncols()]);
        pts_3d
            .slice_mut(s![..2, ..])
            .assign(&(rotation * centered.slice(s![..2, ..])));

        // Vertical scale
        let scale_v =
            &(&pts_3d.slice(s![27..30, 1]) - &pts_3d.slice(s![28..31, 1])) / &self.base_scale_v;
        pts_3d
            .slice_mut(s![.., 1])
            .div_assign(scale_v.mean().unwrap());

        // Horizontal scale
        pts_3d.slice_mut(s![0, ..]).div_assign(
            &((&self.face_3d.select(Axis(0), &[0, 36, 42]).slice(s![.., 0])
                - &self.face_3d.select(Axis(0), &[16, 39, 45]).slice(s![.., 0]))
                .mapv(|x| x.abs())
                / &self.base_scale_h),
        );

        pts_3d
    }
}
