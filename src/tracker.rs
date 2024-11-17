use glam::Vec2;
use ndarray::{array, s, stack, Array1, Array2, Array3, Array4, Axis, Ix3, Ix4};
use ndarray_linalg::solve::Inverse;
use opencv::{
    calib3d::{rodrigues, solve_pnp, SOLVEPNP_ITERATIVE},
    core::{no_array, Mat, Point2f, Point3f, Rect, Scalar, Size, Vec3b, BORDER_CONSTANT, CV_32F},
    imgproc::{get_rotation_matrix_2d, warp_affine, INTER_LINEAR},
    prelude::*,
};
use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder};

use std::ops::Sub;
use std::{f32::consts::PI, path::Path};

use crate::{
    face_info::FaceInfo,
    math::{compensate_rotation, matrix_to_quaternion, rotate},
};

fn mat_to_array2(mat: &Mat) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let rows = mat.rows();
    let cols = mat.cols();
    let data: Vec<f32> = mat.data_typed()?.to_vec();
    Array2::from_shape_vec((rows as usize, cols as usize), data).map_err(|e| e.into())
}

pub struct Tracker {
    // Session and model related
    session: Session,
    gaze_model: Session,
    input_name: String,

    // Camera parameters
    pub camera: Array2<f32>,         // Shape: (3, 3)
    pub inverse_camera: Array2<f32>, // Shape: (3, 3)
    pub dist_coeffs: Array2<f32>,    // Shape: (4, 1)

    // Face model and tracking
    pub face_3d: Array2<f32>,                            // Shape: (70, 3)
    pub face_bounding_box: Option<(f32, f32, f32, f32)>, // (x, y, width, height)

    // Frame info
    pub frame_count: i32,
    pub width: i32,
    pub height: i32,

    // Configuration
    pub threshold: f32,
    pub max_threads: usize,
    pub bbox_growth: f32,
    pub no_gaze: bool,
    pub debug_gaze: bool,
    pub feature_level: i32,
    pub max_feature_updates: i32,
    pub static_model: bool,
}

impl Tracker {
    pub fn new(
        width: i32,
        height: i32,
        model_type: i32,
        bbox_growth: f32,
        max_threads: usize,
        model_dir: Option<&Path>,
        no_gaze: bool,
        max_feature_updates: i32,
        static_model: bool,
        feature_level: i32,
        environment: Environment,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Set up model paths
        let models = vec![
            "lm_model0_opt.onnx",
            "lm_model1_opt.onnx",
            "lm_model2_opt.onnx",
            "lm_model3_opt.onnx",
            "lm_model4_opt.onnx",
        ];
        let model_path = model_dir
            .unwrap_or(Path::new("models"))
            .join(&models[model_type as usize]);
        let gaze_model_path = model_dir
            .unwrap_or(Path::new("models"))
            .join("mnv3_gaze32_split_opt.onnx");

        // Create sessions
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(std::cmp::min(max_threads, 4))
            .unwrap()
            .commit_from_file(&model_path)
            .unwrap();

        let gaze_model = Session::builder()
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .commit_from_file(&gaze_model_path)
            .unwrap();

        // Get input name
        let input_name = session.inputs[0].name.clone();

        // Initialize camera matrix
        let camera = Array2::from_shape_vec(
            (3, 3),
            vec![
                width as f32,
                0.0,
                width as f32 / 2.0,
                0.0,
                width as f32,
                height as f32 / 2.0,
                0.0,
                0.0,
                1.0,
            ],
        )?;

        let inverse_camera = camera.inv()?;
        let dist_coeffs = Array2::zeros((4, 1));

        // Initialize face 3D model (you'll need to add the actual values)
        let face_3d = Array2::zeros((70, 3)); // Add actual face 3D model values here

        Ok(Self {
            session,
            gaze_model,
            input_name,
            camera,
            inverse_camera,
            dist_coeffs,
            face_3d,
            face_bounding_box: None,
            frame_count: 0,
            width,
            height,
            threshold: 0.6,
            max_threads,
            bbox_growth,
            no_gaze,
            debug_gaze: false,
            feature_level,
            max_feature_updates,
            static_model,
        })
    }

    pub fn predict(&mut self, frame: &Mat) -> Option<FaceInfo> {
        self.frame_count += 1;

        // Get confidence and landmarks
        let (confidence, landmarks) = match self.get_confidence_and_landmarks(
            self.face_bounding_box
                .unwrap_or((0.0, 0.0, self.width as f32, self.height as f32)),
            frame,
        ) {
            Ok((conf, lms)) => (conf?, lms?),
            Err(_) => return None,
        };

        if confidence <= self.threshold {
            self.face_bounding_box = None;
            return None;
        }

        // Create and update FaceInfo
        let mut face_info = FaceInfo::new(0, &self);

        // Get eye state
        let eye_state = match self.get_eye_state(frame, &landmarks) {
            Ok(state) => state,
            Err(_) => Array2::from_shape_vec((2, 4), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                .unwrap(),
        };

        // Update face info fields
        face_info.conf = Some(confidence);
        face_info.lms = Some(landmarks.clone());
        face_info.eye_state = Some(eye_state);
        face_info.coord = Some(landmarks.slice(s![.., ..2]).mean_axis(Axis(0))?);
        face_info.alive = true;
        face_info.frame_count = self.frame_count;

        // Estimate depth and adjust 3D model
        self.estimate_depth(&mut face_info).unwrap();
        face_info.adjust_3d();

        // Update bounding box
        if let Some(lms) = &face_info.lms {
            let landmarks = lms.slice(s![..66, ..2]);
            if !landmarks.iter().any(|&x| x.is_nan()) {
                let min_coords =
                    landmarks.map_axis(Axis(0), |view| view.fold(f32::INFINITY, |a, &b| a.min(b)));
                let max_coords = landmarks.map_axis(Axis(0), |view| {
                    view.fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                });
                let bbox = (
                    min_coords[1],                 // y1
                    min_coords[0],                 // x1
                    max_coords[1] - min_coords[1], // height
                    max_coords[0] - min_coords[0], // width
                );
                face_info.bbox = Some(bbox);
                self.face_bounding_box = Some(bbox);
                return Some(face_info);
            }
        }

        None
    }

    fn landmarks(
        &self,
        tensor: &Array3<f32>,
        crop_info: (f32, f32, f32, f32),
    ) -> Result<(f32, Array2<f32>), Box<dyn std::error::Error>> {
        let (crop_x1, crop_y1, scale_x, scale_y) = crop_info;
        const TENSOR_DIM: usize = 28;
        const RESOLUTION: f32 = 223.0; // RESOLUTION - 1
        const FACIAL_LANDMARKS_COUNT: usize = 70;

        let landmarks_range = 0..FACIAL_LANDMARKS_COUNT;
        let x_offsets_range = FACIAL_LANDMARKS_COUNT..(FACIAL_LANDMARKS_COUNT * 2);
        let y_offsets_range = (FACIAL_LANDMARKS_COUNT * 2)..(FACIAL_LANDMARKS_COUNT * 3);

        // Reshape and get maxima
        let tensor_main = tensor
            .slice(s![landmarks_range, .., ..])
            .into_shape([FACIAL_LANDMARKS_COUNT, TENSOR_DIM * TENSOR_DIM])?;

        let mut tensor_maxima = Array1::zeros(FACIAL_LANDMARKS_COUNT);
        let mut tensor_confidence = Array1::zeros(FACIAL_LANDMARKS_COUNT);

        // Find maxima and confidence values
        for i in 0..FACIAL_LANDMARKS_COUNT {
            let max_idx = tensor_main
                .slice(s![i, ..])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            tensor_maxima[i] = max_idx as f32;
            tensor_confidence[i] = tensor_main[[i, max_idx]];
        }

        // Get x and y offsets
        let tensor_x = tensor
            .slice(s![x_offsets_range, .., ..])
            .into_shape([FACIAL_LANDMARKS_COUNT, TENSOR_DIM * TENSOR_DIM])?;
        let tensor_y = tensor
            .slice(s![y_offsets_range, .., ..])
            .into_shape([FACIAL_LANDMARKS_COUNT, TENSOR_DIM * TENSOR_DIM])?;

        let mut tensor_offset_x = Array1::zeros(FACIAL_LANDMARKS_COUNT);
        let mut tensor_offset_y = Array1::zeros(FACIAL_LANDMARKS_COUNT);

        for i in 0..FACIAL_LANDMARKS_COUNT {
            let idx = tensor_maxima[i] as usize;
            tensor_offset_x[i] = tensor_x[[i, idx]];
            tensor_offset_y[i] = tensor_y[[i, idx]];
        }

        // Apply logistic function and scale
        tensor_offset_x.mapv_inplace(|x| RESOLUTION * logit(x, 16.0));
        tensor_offset_y.mapv_inplace(|x| RESOLUTION * logit(x, 16.0));

        // Calculate final coordinates
        let tensor_x = crop_y1
            + scale_y
                * (RESOLUTION
                    * (tensor_maxima.mapv(|x| (x / TENSOR_DIM as f32).floor())
                        / (TENSOR_DIM as f32 - 1.0))
                    + &tensor_offset_x);
        let tensor_y = crop_x1
            + scale_x
                * (RESOLUTION
                    * (tensor_maxima.mapv(|x| (x % TENSOR_DIM as f32).floor())
                        / (TENSOR_DIM as f32 - 1.0))
                    + &tensor_offset_y);

        // Stack results
        let mut landmarks = Array2::zeros((FACIAL_LANDMARKS_COUNT, 3));
        for i in 0..FACIAL_LANDMARKS_COUNT {
            landmarks[[i, 0]] = tensor_x[i];
            landmarks[[i, 1]] = tensor_y[i];
            landmarks[[i, 2]] = tensor_confidence[i];
        }

        // Handle NaN values
        for i in 0..FACIAL_LANDMARKS_COUNT {
            if landmarks.row(i).iter().any(|&x| x.is_nan()) {
                landmarks.row_mut(i).fill(0.0);
            }
        }

        Ok((tensor_confidence.mean().unwrap(), landmarks))
    }

    fn get_confidence_and_landmarks(
        &self,
        face_bounding_box: (f32, f32, f32, f32),
        image: &Mat,
    ) -> Result<(Option<f32>, Option<Array2<f32>>), Box<dyn std::error::Error>> {
        const RESOLUTION: i32 = 224;
        const MEAN_224: f32 = 0.485;
        const STD_224: f32 = 0.456;

        let (x, y, width, height) = face_bounding_box;

        let crop_x1 = (x - width * 0.1) as i32;
        let crop_y1 = (y - height * 0.125) as i32;
        let crop_x2 = (x + width + width * 0.1) as i32;
        let crop_y2 = (y + height + height * 0.125) as i32;

        // Clamp to image boundaries
        let crop_x1 = crop_x1.max(0).min(self.width);
        let crop_y1 = crop_y1.max(0).min(self.height);
        let crop_x2 = crop_x2.max(0).min(self.width);
        let crop_y2 = crop_y2.max(0).min(self.height);

        let scale_x = (crop_x2 - crop_x1) as f32 / RESOLUTION as f32;
        let scale_y = (crop_y2 - crop_y1) as f32 / RESOLUTION as f32;

        if crop_x2 - crop_x1 >= 4 && crop_y2 - crop_y1 >= 4 {
            // Crop and convert image
            let roi = Mat::roi(
                image,
                opencv::core::Rect::new(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1),
            )?;

            let mut resized = unsafe {
                Mat::new_size(Size::new(RESOLUTION, RESOLUTION), opencv::core::CV_32FC3)
            }?;
            opencv::imgproc::resize(
                &roi,
                &mut resized,
                Size::new(RESOLUTION, RESOLUTION),
                0.0,
                0.0,
                opencv::imgproc::INTER_LINEAR,
            )?;

            // Convert to RGB float32 array and normalize
            let mut float_img = Mat::new_size_with_default(
                Size::new(RESOLUTION, RESOLUTION),
                opencv::core::CV_32FC3,
                Scalar::all(0.0),
            )?;
            resized.convert_to(
                &mut float_img,
                opencv::core::CV_32F,
                STD_224.into(),
                MEAN_224.into(),
            )?;

            // Convert to ONNX input format
            let mut input_tensor =
                Array4::<f32>::zeros((1, 3, RESOLUTION as usize, RESOLUTION as usize));
            for y in 0..RESOLUTION {
                for x in 0..RESOLUTION {
                    let pixel = float_img.at_2d::<opencv::core::Vec3f>(y, x)?;
                    input_tensor[[0, 0, y as usize, x as usize]] = pixel[2]; // B
                    input_tensor[[0, 1, y as usize, x as usize]] = pixel[1]; // G
                    input_tensor[[0, 2, y as usize, x as usize]] = pixel[0]; // R
                }
            }

            // Run inference
            let output = self
                .session
                .run(ort::inputs!("input" => input_tensor).unwrap())?;
            let output_tensor = output["output"].try_extract_tensor::<f32>()?;

            // Process landmarks
            let (confidence, landmarks) = self.landmarks(
                &output_tensor
                    .view()
                    .into_dimensionality::<Ix3>()?
                    .to_owned(),
                (crop_x1 as f32, crop_y1 as f32, scale_x, scale_y),
            )?;

            Ok((Some(confidence), Some(landmarks)))
        } else {
            Ok((None, None))
        }
    }

    pub fn get_eye_state(
        &self,
        frame: &Mat,
        lms: &Array2<f32>,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        if self.no_gaze {
            return Ok(Array2::from_shape_vec(
                (2, 4),
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            )?);
        }

        // Extract face frame
        let (face_frame, lms_adjusted, offset) = self.extract_face(frame, lms)?;

        // Prepare eyes
        let right_eye = self.prepare_eye(
            &face_frame,
            frame,
            &array![
                [lms_adjusted[[36, 0]], lms_adjusted[[36, 1]]],
                [lms_adjusted[[39, 0]], lms_adjusted[[39, 1]]]
            ],
            false,
        )?;

        let left_eye = self.prepare_eye(
            &face_frame,
            frame,
            &array![
                [lms_adjusted[[42, 0]], lms_adjusted[[42, 1]]],
                [lms_adjusted[[45, 0]], lms_adjusted[[45, 1]]]
            ],
            true,
        )?;

        // Check if eye preparation failed
        if right_eye.0.is_none() || left_eye.0.is_none() {
            return Ok(Array2::from_shape_vec(
                (2, 4),
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            )?);
        }

        // Combine eye tensors for model input
        let both_eyes = ndarray::concatenate(
            Axis(0),
            &[right_eye.0.unwrap().view(), left_eye.0.unwrap().view()],
        )?;

        // Run gaze model inference
        let output = self
            .gaze_model
            .run(ort::inputs!("input" => both_eyes).unwrap())?;
        let results = output["output"].try_extract_tensor::<f32>()?.to_owned();
        let results = results.view().into_dimensionality::<Ix4>()?;

        let mut eye_state = Array2::<f32>::zeros((2, 4));

        // Process each eye
        for i in 0..2 {
            let result_slice = results.slice(s![i, .., .., ..]);

            // Find maximum confidence position
            let mut max_conf = f32::MIN;
            let mut max_x = 0;
            let mut max_y = 0;

            for x in 0..8 {
                for y in 0..8 {
                    let conf = result_slice[[0, x, y]];
                    if conf > max_conf {
                        max_conf = conf;
                        max_x = x;
                        max_y = y;
                    }
                }
            }

            // Calculate eye position
            let off_x = 32.0 * logit(result_slice[[1, max_x, max_y]], 16.0);
            let off_y = 32.0 * logit(result_slice[[2, max_x, max_y]], 16.0);

            let mut eye_x = 32.0 * max_x as f32 / 8.0 + off_x;
            let eye_y = 32.0 * max_y as f32 / 8.0 + off_y;

            // Adjust x coordinate for left eye
            if i == 1 {
                eye_x = 32.0 - eye_x;
            }

            // Transform coordinates back to original image space
            let (e_x, e_y, scale, reference, angle) = if i == 0 {
                (
                    right_eye.1,
                    right_eye.2,
                    right_eye.3.clone(),
                    right_eye.4,
                    right_eye.5,
                )
            } else {
                (
                    left_eye.1,
                    left_eye.2,
                    left_eye.3.clone(),
                    left_eye.4,
                    left_eye.5,
                )
            };

            let mut transformed_x = e_x + scale[0] * eye_x;
            let mut transformed_y = e_y + scale[1] * eye_y;

            // Rotate point back
            let (rotated_x, rotated_y) = rotate(
                Vec2::new(reference.x, reference.y),
                Vec2::new(transformed_x, transformed_y),
                -angle,
            )
            .into();

            transformed_x = rotated_x + offset[0];
            transformed_y = rotated_y + offset[1];

            // Store results
            eye_state[[i, 0]] = 1.0; // Open state
            eye_state[[i, 1]] = transformed_y;
            eye_state[[i, 2]] = transformed_x;
            eye_state[[i, 3]] = max_conf;
        }

        // Handle NaN values
        for mut row in eye_state.rows_mut() {
            if row.iter().any(|&x| x.is_nan()) {
                row.assign(&array![1.0, 0.0, 0.0, 0.0]);
            }
        }

        Ok(eye_state)
    }

    fn extract_face(
        &self,
        frame: &Mat,
        lms: &Array2<f32>,
    ) -> Result<(Mat, Array2<f32>, Array1<f32>), Box<dyn std::error::Error>> {
        let landmarks = lms.slice(s![.., ..2]);

        // Find bounding box
        let min_coords =
            landmarks.map_axis(Axis(0), |view| view.fold(f32::INFINITY, |a, &b| a.min(b)));
        let max_coords = landmarks.map_axis(Axis(0), |view| {
            view.fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        });

        let radius_x = 1.2 * (max_coords[0] - min_coords[0]) / 2.0;
        let radius_y = 1.2 * (max_coords[1] - min_coords[1]) / 2.0;
        let center = (min_coords + max_coords) / 2.0;

        // Calculate crop coordinates
        let x1 = (center[0] - radius_x).max(0.0).min(frame.cols() as f32) as i32;
        let y1 = (center[1] - radius_y).max(0.0).min(frame.rows() as f32) as i32;
        let x2 = (center[0] + radius_x + 1.0)
            .max(0.0)
            .min(frame.cols() as f32) as i32;
        let y2 = (center[1] + radius_y + 1.0)
            .max(0.0)
            .min(frame.rows() as f32) as i32;

        // Crop frame
        let roi = Mat::roi(frame, Rect::new(x1, y1, x2 - x1, y2 - y1))?.try_clone()?;

        // Adjust landmarks
        let offset = Array1::from_vec(vec![x1 as f32, y1 as f32]);
        let adjusted_lms = (&landmarks - &offset).mapv(|x| x as f32);

        Ok((roi, adjusted_lms, offset))
    }

    fn prepare_eye(
        &self,
        frame: &Mat,
        full_frame: &Mat,
        corners: &Array2<f32>,
        flip: bool,
    ) -> Result<(Option<Array4<f32>>, f32, f32, Vec<f32>, Point2f, f32), Box<dyn std::error::Error>>
    {
        const STD_32: f32 = 0.456;
        const MEAN_32: f32 = 0.485;

        let (c1, c2) = (
            Vec2::new(corners[[0, 0]], corners[[0, 1]]),
            Vec2::new(corners[[1, 0]], corners[[1, 1]]),
        );

        let (c2_comp, angle) = compensate_rotation(c1, c2);
        let center = Point2f::new((c1.x + c2_comp.x) / 2.0, (c1.y + c2_comp.y) / 2.0);

        let radius = ((c1.x - c2_comp.x).powi(2) + (c1.y - c2_comp.y).powi(2)).sqrt() / 2.0;
        let scale = vec![radius * 1.4 / 32.0, radius * 1.2 / 32.0];

        let h = frame.rows() as f32;
        let w = frame.cols() as f32;

        // Calculate crop coordinates
        let x1 = (center.x - scale[0]).max(0.0).min(w);
        let y1 = (center.y - scale[1]).max(0.0).min(h);
        let x2 = (center.x + scale[0]).max(0.0).min(w);
        let y2 = (center.y + scale[1]).max(0.0).min(h);

        if (x2 - x1) < 1.0 || (y2 - y1) < 1.0 {
            return Ok((None, 0.0, 0.0, scale, Point2f::new(c1.x, c1.y), angle));
        }

        // Rotate and crop image
        let rot_mat = get_rotation_matrix_2d(center, (angle * 180.0 / PI).into(), 1.0)?;
        let mut rotated = Mat::default();
        warp_affine(
            frame,
            &mut rotated,
            &rot_mat,
            frame.size()?,
            INTER_LINEAR,
            BORDER_CONSTANT,
            Scalar::all(0.0),
        )?;

        let mut eye_region = Mat::roi(
            &rotated,
            Rect::new(x1 as i32, y1 as i32, (x2 - x1) as i32, (y2 - y1) as i32),
        )?
        .try_clone()?;

        if flip {
            opencv::core::flip(&eye_region.clone(), &mut eye_region, 1)?;
        }

        // Resize to 32x32
        let mut resized = Mat::default();
        opencv::imgproc::resize(
            &eye_region,
            &mut resized,
            Size::new(32, 32),
            0.0,
            0.0,
            INTER_LINEAR,
        )?;

        // Convert to tensor
        let mut tensor = Array4::<f32>::zeros((1, 3, 32, 32));
        for y in 0..32 {
            for x in 0..32 {
                let pixel = resized.at_2d::<Vec3b>(y, x)?;
                for c in 0..3 {
                    tensor[[0, c, y as usize, x as usize]] = (pixel[c] as f32) * STD_32 + MEAN_32;
                }
            }
        }

        Ok((Some(tensor), x1, y1, scale, Point2f::new(c1.x, c1.y), angle))
    }

    fn estimate_depth(&self, face_info: &mut FaceInfo) -> Result<(), Box<dyn std::error::Error>> {
        // Create landmarks array with eye state
        let mut lms = Array2::zeros((70, 3));
        lms.slice_mut(s![..66, ..])
            .assign(&face_info.lms.as_ref().unwrap());
        if let Some(eye_state) = &face_info.eye_state {
            for i in 0..2 {
                lms[[66 + i, 0]] = eye_state[[i, 2]]; // x
                lms[[66 + i, 1]] = eye_state[[i, 1]]; // y
                lms[[66 + i, 2]] = eye_state[[i, 3]]; // confidence
            }
        }

        let object_pts: opencv::core::Vector<Point3f> = face_info
            .contour
            .rows()
            .into_iter()
            .map(|i| Point3f::new(i[0], i[1], i[2]))
            .collect();

        let image_pts: opencv::core::Vector<Point2f> = face_info
            .contour_pts
            .iter()
            .map(|&idx| Point2f::new(lms[[idx, 0]], lms[[idx, 1]]))
            .collect();

        // Convert camera matrix and distortion coefficients
        let camera_mat = Mat::from_slice(&self.camera.as_slice().unwrap())?;
        let dist_coeffs: opencv::core::Vector<f32> = opencv::core::Vector::from_iter(self.dist_coeffs.iter().cloned());

        // Solve PnP
        let (success, rotation, translation) = if let Some(rot) = &face_info.rotation {
            let mut rvec = Mat::from_slice(&rot.to_vec())?.try_clone()?;
            let mut tvec =
                Mat::from_slice(&face_info.translation.as_ref().unwrap().to_vec())?.try_clone()?;

            let success = solve_pnp(
                &object_pts,
                &image_pts,
                &camera_mat,
                &dist_coeffs,
                &mut rvec,
                &mut tvec,
                true,
                SOLVEPNP_ITERATIVE,
            )?;

            (success, rvec, tvec)
        } else {
            let mut rvec = Mat::zeros(3, 1, CV_32F)?.to_mat()?;
            let mut tvec = Mat::zeros(3, 1, CV_32F)?.to_mat()?;

            let success = solve_pnp(
                &object_pts,
                &image_pts,
                &camera_mat,
                &dist_coeffs,
                &mut rvec,
                &mut tvec,
                true,
                SOLVEPNP_ITERATIVE,
            )?;

            (success, rvec, tvec)
        };

        let mut pts_3d = Array2::zeros((70, 3));

        if !success {
            face_info.rotation = Some(array![0.0, 0.0, 0.0]);
            face_info.translation = array![0.0, 0.0, 0.0].into();
            face_info.success = Some(false);
            face_info.quaternion = Some(array![0.0, 0.0, 0.0, 0.0]);
            face_info.euler = Some(array![0.0, 0.0, 0.0]);
            face_info.pnp_error = 99999.0;
            face_info.pts_3d = Some(pts_3d);
            face_info.lms = Some(lms);
            return Ok(());
        }

        // Convert rotation vector to matrix and get inverse
        let mut rmat = Mat::default();
        rodrigues(&rotation, &mut rmat, &mut no_array())?;
        let rmat_array = mat_to_array2(&rmat)?;
        let inverse_rotation = rmat_array.inv()?;

        // Calculate reference points
        let t_reference = face_info.face_3d.dot(&rmat_array.t());
        let t_reference = &t_reference + &face_info.translation.as_ref().unwrap().view();
        let t_reference = t_reference.dot(&self.camera.t());

        let mut t_depth = t_reference.slice(s![.., 2]).to_owned();
        t_depth.mapv_inplace(|x| if x == 0.0 { 0.000001 } else { x });

        let t_depth_e = t_depth.insert_axis(Axis(1));
        let t_reference = &t_reference / &t_depth_e;

        // Calculate 3D points for first 66 landmarks
        let lms_stack = stack![
            Axis(1),
            lms.slice(s![..66, 0]),
            lms.slice(s![..66, 1]),
            Array1::ones(66)
        ];
        pts_3d.slice_mut(s![..66, ..]).assign(
            &((&lms_stack * &t_depth_e.slice(s![..66, ..]))
                .into_shape((66, 3))?
                .dot(&self.inverse_camera.t())
                .sub(&face_info.translation.as_ref().unwrap().view())
                .dot(&inverse_rotation.t())),
        );

        // Calculate initial PnP error
        let mut pnp_error = (&lms.slice(s![..17, ..2]) - &t_reference.slice(s![..17, ..2]))
            .mapv(|x| x.powi(2))
            .sum();
        pnp_error += (&lms.slice(s![30..31, ..2]) - &t_reference.slice(s![30..31, ..2]))
            .mapv(|x| x.powi(2))
            .sum();

        if pnp_error.is_nan() {
            pnp_error = 9999999.0;
        }

        // Process remaining points (eyes)
        for i in 0..4 {
            if i == 2 {
                // Right eyeball
                let eye_center = (&pts_3d.row(36) + &pts_3d.row(39)) / 2.0;
                let d_corner = ((&pts_3d.row(36) - &pts_3d.row(39)).mapv(|x| x.powi(2)))
                    .sum()
                    .sqrt();
                let depth = 0.385 * d_corner;
                pts_3d.row_mut(68).assign(&array![
                    eye_center[0],
                    eye_center[1],
                    eye_center[2] - depth
                ]);
                continue;
            }
            if i == 3 {
                // Left eyeball
                let eye_center = (&pts_3d.row(42) + &pts_3d.row(45)) / 2.0;
                let d_corner = ((&pts_3d.row(42) - &pts_3d.row(45)).mapv(|x| x.powi(2)))
                    .sum()
                    .sqrt();
                let depth = 0.385 * d_corner;
                pts_3d.row_mut(69).assign(&array![
                    eye_center[0],
                    eye_center[1],
                    eye_center[2] - depth
                ]);
                continue;
            }

            let (d1, d2, pt) = if i == 0 {
                let d1 = ((&lms.row(66).slice(s![..2]) - &lms.row(36).slice(s![..2]))
                    .mapv(|x| x.powi(2)))
                .sum()
                .sqrt();
                let d2 = ((&lms.row(66).slice(s![..2]) - &lms.row(39).slice(s![..2]))
                    .mapv(|x| x.powi(2)))
                .sum()
                .sqrt();
                let pt = (&pts_3d.row(36) * d1 + &pts_3d.row(39) * d2) / (d1 + d2);
                (d1, d2, pt)
            } else {
                let d1 = ((&lms.row(67).slice(s![..2]) - &lms.row(42).slice(s![..2]))
                    .mapv(|x| x.powi(2)))
                .sum()
                .sqrt();
                let d2 = ((&lms.row(67).slice(s![..2]) - &lms.row(45).slice(s![..2]))
                    .mapv(|x| x.powi(2)))
                .sum()
                .sqrt();
                let pt = (&pts_3d.row(42) * d1 + &pts_3d.row(45) * d2) / (d1 + d2);
                (d1, d2, pt)
            };

            if i < 2 {
                let mut reference = rmat_array.dot(&pt);
                reference = reference + &face_info.translation.as_ref().unwrap().view();
                reference = self.camera.dot(&reference);
                let depth = reference[2];
                let mut pt_3d = array![lms[[66 + i, 0]] * depth, lms[[66 + i, 1]] * depth, depth];
                pt_3d = self.inverse_camera.dot(&pt_3d);
                pt_3d = pt_3d - &face_info.translation.as_ref().unwrap().view();
                pt_3d = inverse_rotation.dot(&pt_3d);
                pts_3d.row_mut(66 + i).assign(&pt_3d);
            }
        }

        // Handle NaN values
        for mut row in pts_3d.rows_mut() {
            if row.iter().any(|&x| x.is_nan()) {
                row.fill(0.0);
            }
        }

        // Calculate final PnP error
        let pnp_error = (pnp_error / (2.0 * image_pts.len() as f32)).sqrt();

        // Handle high error cases
        if pnp_error > 300.0 {
            face_info.fail_count += 1;
            if face_info.fail_count > 5 {
                println!(
                    "Detected anomaly when 3D fitting face {}. Resetting.",
                    face_info.id
                );
                face_info.face_3d = self.face_3d.clone();
                face_info.rotation = None;
                face_info.translation = array![0.0, 0.0, 0.0].into();
                face_info.update_counts = Array2::zeros((66, 2));
                face_info.update_contour();
            }
        } else {
            face_info.fail_count = 0;
        }

        // Get euler angles
        let euler = decompose_rotation_matrix(&rmat_array)?;
        let rotation_array: [[f32; 3]; 3] = [
            [rmat_array[[0, 0]], rmat_array[[0, 1]], rmat_array[[0, 2]]],
            [rmat_array[[1, 0]], rmat_array[[1, 1]], rmat_array[[1, 2]]],
            [rmat_array[[2, 0]], rmat_array[[2, 1]], rmat_array[[2, 2]]],
        ];
        let quat = matrix_to_quaternion(rotation_array);

        face_info.success = Some(true);
        face_info.quaternion = Some(Array1::from_vec(quat.to_vec()));
        face_info.euler = Some(euler);
        face_info.pnp_error = pnp_error;
        face_info.pts_3d = Some(pts_3d);
        face_info.lms = Some(lms);

        Ok(())
    }
}

fn logit(mut p: f32, factor: f32) -> f32 {
    if p >= 1.0 {
        p = 0.9999999;
    }
    if p <= 0.0 {
        p = 0.0000001;
    }
    p = p / (1.0 - p);
    p.ln() / factor
}

fn decompose_rotation_matrix(r: &Array2<f32>) -> Result<Array1<f32>, Box<dyn std::error::Error>> {
    let sy = (r[[0, 0]].powi(2) + r[[1, 0]].powi(2)).sqrt();

    let (pitch, yaw, roll) = if sy > 1e-6 {
        (
            (-r[[2, 0]]).atan2(sy),
            r[[1, 0]].atan2(r[[0, 0]]),
            r[[2, 1]].atan2(r[[2, 2]]),
        )
    } else {
        ((-r[[2, 0]]).atan2(sy), 0.0, r[[1, 2]].atan2(r[[1, 1]]))
    };

    Ok(array![pitch, yaw, roll])
}
