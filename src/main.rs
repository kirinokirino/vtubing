use ndarray::Array2;
use ort::{GraphOptimizationLevel, Session};
extern crate opencv;
use opencv::{
    core::{Rect2f, Size2i, Vec3f, VecN, CV_32F, CV_32FC1, CV_32FC3},
    imgproc,
    prelude::*,
    videoio::{
        VideoCapture, VideoCaptureTrait, CAP_ANY, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
    },
    Result,
};

pub fn main() {
    let path = "/home/k/Documents/Rust/k/vtubing/onnx/lib/libonnxruntime.so.1.20.0";
    ort::init_from(path).commit().unwrap();

    let model = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .commit_from_file("models/lm_model3_opt.onnx")
        .unwrap();

    // Open the video capture (0 for default webcam, or a video file path)
    let mut capture = VideoCapture::new(0, CAP_ANY).unwrap(); // 0 for default webcam

    // Set the capture resolution to 640x480
    capture.set(CAP_PROP_FRAME_WIDTH, 640.0).unwrap();
    capture.set(CAP_PROP_FRAME_HEIGHT, 480.0).unwrap();

    if !capture.is_opened().unwrap() {
        panic!("Error: Couldn't open video capture.");
    }

    let std_224 = [0.01712475, 0.017507, 0.01742919];
    let mean_224 = [-2.117904, -2.0357141, -1.8044444];
    let mut frame = Mat::default();
    let mut resized = Mat::default();
    let mut rgb = Mat::default();
    let mut normalized = Mat::default();
    loop {
        // Capture a frame from the video capture
        capture.read(&mut frame).unwrap();
        if frame.empty() {
            eprintln!("Error: No frame captured.");
            continue;
        }

        imgproc::resize(
            &frame,
            &mut resized,
            Size2i::new(224, 224),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )
        .unwrap();

        // Step 2: Convert from BGR to RGB by reversing channels
        imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0).unwrap();

        // Step 3: Normalize the image by applying the mean and std deviation
        // We need to cast the Mat to a type that can hold the resulting floating point values (CV_32F)
        rgb.convert_to(&mut normalized, CV_32FC1, 1.0, 0.0).unwrap();

        // Normalize using (pixel - mean) / std
        for row in 0..normalized.rows() {
            for col in 0..normalized.cols() {
                let pixel: &mut VecN<f32, 3> = normalized.at_2d_mut(row, col).unwrap();
                for channel in 0..3usize {
                    pixel[channel] = (pixel[channel] - mean_224[channel]) / std_224[channel];
                }
            }
        }

        // Convert `Mat` to `ndarray::Array4`
        let array = mat_to_ndarray(&normalized);
        let array = array.unwrap();

        // Step 1: Expand dimensions (add a batch dimension)
        let expanded = array.insert_axis(ndarray::Axis(0)); // Adds a batch dimension at axis 0

        // Step 2: Transpose to change shape from (batch_size, height, width, channels)
        // to (batch_size, channels, height, width)
        let transposed = expanded.permuted_axes([0, 1, 2, 3]);

        let inputs = transposed;
        let outputs = model.run(ort::inputs!("input" => inputs).unwrap()).unwrap();
        let predictions = outputs["output0"].try_extract_tensor::<f32>().unwrap();
        //ort::inputs![]
        // let outputs = model.run(ort::inputs!["image" => image]?)?;
        // let predictions = outputs["output0"].try_extract_tensor::<f32>()?;

        println!("Frame captured: {:?}", frame);
    }

    // Close the video capture and window
    capture.release().unwrap();
}

// Convert a Mat (OpenCV) to ndarray (Rust)
fn mat_to_ndarray(mat: &Mat) -> Result<ndarray::Array3<f32>> {
    assert!(mat.typ() == CV_32FC3);
    // Get the number of rows, columns, and channels
    let rows = mat.rows();
    let cols = mat.cols();
    let channels = mat.channels();

    let mat_data = mat.data_typed::<Vec3f>().unwrap();
    let flattened_data: Vec<f32> = mat_data
        .iter()
        .flat_map(|vec| vec.0) // `vec.0` accesses the internal `[f32; 3]` in Vec3f
        .collect();

    // Create the ndarray Array4 (batch, channels, height, width)
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

/// Function to convert an OpenCV `Mat` to an `ndarray::Array2`
fn mat_to_array2(mat: &Mat) -> Result<Array2<f32>> {
    // Ensure the Mat is of the expected type (CV_32F)
    if mat.typ() != CV_32F {
        return Err(opencv::Error::new(
            opencv::core::StsBadArg,
            format!(
                "Mat type must be CV_32F for conversion to Array2, found: {}",
                mat.typ()
            ),
        ));
    }

    // Get the number of rows and columns from the Mat
    let rows = mat.rows();
    let cols = mat.cols();

    // Get a flat slice of the Mat's data
    let mat_data = mat.data_typed::<f32>().unwrap();

    // Convert the flat data into an ndarray::Array2
    let array = Array2::<f32>::from_shape_vec((rows as usize, cols as usize), mat_data.to_vec());

    match array {
        Ok(array) => Ok(array),
        Err(err) => Err(opencv::Error {
            code: opencv::core::StsError,
            message: format!("ShapeError: {err}"),
        }),
    }
}
