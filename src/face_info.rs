use glam::Vec2;
use ndarray::{s, Array1, Array2, ArrayView1, Axis};

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
                base_scale_v.zip_mut_with(&tracker.face_3d.slice(s![28..31, 1]), |a, b| *a -= *b);
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

    pub fn adjust_3d(&mut self) {
        // Early return conditions
        if self.conf.unwrap_or(0.0) < 0.4 {
            //|| self.pnp_error > 300.0 {
            return;
        }

        let max_runs = 1;
        let mut eligible: Vec<usize> = (0..66).filter(|&x| x != 30).collect();
        let mut changed_any = false;
        let mut update_type: i32 = -1;
        let mut d_o: Array1<f32> = Array1::ones(66);
        let mut d_c: Array1<f32> = Array1::ones(66);

        for runs in 0..max_runs {
            // Create random adjustments (1.0 +/- 0.01)
            let mut r = Array2::zeros((66, 3));
            for i in 0..66 {
                for j in 0..3 {
                    r[[i, j]] = 1.0 + (fastrand::f32() * 0.02 - 0.01);
                }
            }
            r.slice_mut(s![30, ..]).fill(1.0);

            let euler = self.euler.as_ref().unwrap();
            if euler[0] > -165.0 && euler[0] < 145.0 {
                continue;
            } else if euler[1] > -10.0 && euler[1] < 20.0 {
                r.slice_mut(s![.., 2]).fill(1.0);
                update_type = 0;
            } else {
                r.slice_mut(s![.., 0..2]).fill(1.0);
                if euler[2] > 120.0 || euler[2] < 60.0 {
                    continue;
                }

                // Update eligible points based on euler[1]
                if euler[1] < -10.0 {
                    update_type = 1;
                    let fixed_indices = vec![
                        0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 31, 32, 36, 37, 38, 39, 40, 41,
                        48, 49, 56, 57, 58, 59, 65,
                    ];
                    for &idx in &fixed_indices {
                        r[[idx, 2]] = 1.0;
                    }
                    eligible = vec![
                        8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34,
                        35, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64,
                    ];
                } else {
                    update_type = 1;
                    let fixed_indices = vec![
                        9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 34, 35, 42, 43, 44, 45,
                        46, 47, 51, 52, 53, 54, 61, 62, 63,
                    ];
                    for &idx in &fixed_indices {
                        r[[idx, 2]] = 1.0;
                    }
                    eligible = vec![
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 27, 28, 29, 31, 32, 33, 36,
                        37, 38, 39, 40, 41, 48, 49, 50, 55, 56, 57, 58, 59, 60, 64, 65,
                    ];
                }
            }

            if self.limit_3d_adjustment {
                eligible.retain(|&i| {
                    self.update_counts[[i, update_type as usize]]
                        < self.update_counts[[i, (update_type - 1).unsigned_abs() as usize]]
                            + self.update_count_delta
                });

                if eligible.is_empty() {
                    break;
                }
            }

            if runs == 0 {
                let mut updated = self.face_3d.slice(s![0..66, ..]).to_owned();
                let mut o_projected = Array2::ones((66, 2));

                // TODO: Implement cv2.projectPoints equivalent
                // o_projected.slice_mut(s![eligible, ..]) = project_points(
                //     &self.face_3d.select(Axis(0), &eligible),
                //     &self.rotation.unwrap(),
                //     &self.translation.unwrap(),
                //     camera,
                //     dist_coeffs
                // );

                let c = &updated * &r;
                let c_projected = Array2::zeros((66, 2));

                // TODO: Implement cv2.projectPoints equivalent for c_projected

                let mut changed = false;

                // Calculate distances
                for &idx in &eligible {
                    let o_diff = &o_projected.slice(s![idx, ..]).to_owned()
                        - &self.lms.as_ref().unwrap().slice(s![idx, 0..2]).to_owned();
                    let c_diff = &c_projected.slice(s![idx, ..]).to_owned()
                        - &self.lms.as_ref().unwrap().slice(s![idx, 0..2]).to_owned();
                    d_o[idx] = norm2(&o_diff.view());
                    d_c[idx] = norm2(&c_diff.view());
                }

                // Find indices where d_c < d_o
                let indices: Vec<usize> = (0..66).filter(|&i| d_c[i] < d_o[i]).collect();

                if !indices.is_empty() {
                    let indices = if self.limit_3d_adjustment {
                        indices
                            .into_iter()
                            .filter(|&i| eligible.contains(&i))
                            .collect::<Vec<_>>()
                    } else {
                        indices
                    };

                    if !indices.is_empty() {
                        for &idx in &indices {
                            self.update_counts[[idx, update_type as usize]] += 1.0;
                            updated.slice_mut(s![idx, ..]).assign(&c.slice(s![idx, ..]));
                            o_projected
                                .slice_mut(s![idx, ..])
                                .assign(&c_projected.slice(s![idx, ..]));
                        }
                        changed = true;
                    }
                }

                changed_any = changed_any || changed;

                if !changed {
                    break;
                }

                if changed_any {
                    // Update weighted by point confidence
                    let mut weights = Array2::zeros((66, 3));
                    weights.assign(&self.lms.as_ref().unwrap().slice(s![0..66, 2..3]));
                    weights.mapv_inplace(|x| if x > 0.7 { 1.0 } else { x });
                    weights.mapv_inplace(|x| 1.0 - x);

                    let update_indices: Vec<usize> = if self.limit_3d_adjustment {
                        (0..66)
                            .filter(|&i| {
                                self.update_counts[[i, update_type as usize]]
                                    <= self.update_count_max
                            })
                            .collect()
                    } else {
                        (0..66).collect()
                    };

                    for &idx in &update_indices {
                        let copy = self.face_3d.clone();
                        self.face_3d.slice_mut(s![idx, ..]).assign(
                            &(copy.slice(s![idx, ..]).to_owned()
                                * weights.slice(s![idx, ..]).to_owned()
                                + updated.slice(s![idx, ..]).to_owned()
                                    * (1.0 - weights.slice(s![idx, ..]).to_owned())),
                        );
                    }
                    self.update_contour();
                }
            }
        }

        let normalized_pts_3d = self.normalize_pts3d(self.pts_3d.as_ref().unwrap().clone());
        if let Some(pts_3d) = self.pts_3d.as_mut() {
            *pts_3d = normalized_pts_3d;
        }

        // TODO: Implement feature extraction and eye blink calculation
        // This would depend on your FeatureExtractor implementation
    }
}

// Helper function to calculate L2 norm of a 1D array
fn norm2(arr: &ArrayView1<f32>) -> f32 {
    arr.iter().map(|&x| x * x).sum::<f32>().sqrt()
}
