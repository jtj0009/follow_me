use anyhow::Result;
use chrono::Utc;
use geometry_msgs::msg::Twist;
use opencv::{
    core, dnn, imgcodecs, imgproc, prelude::*,
    core::Vector,
};
use rclrs::{Context};
use serde::Deserialize;
use std::env;
use std::fs;
use std::process::Command;
use std::thread::sleep;
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct Config {
    save_thumbnails: bool,
    centroid_threshold: f32,
    area_threshold: f32,
    confidence_threshold: f32,
}

struct Detection {
    centroid: (f32, f32),
    area: f32,
    bbox: core::Rect, // bounding box (x, y, width, height)
}

// Capture an image using rpicam-still.
fn capture_image() -> Result<Mat> {
    Command::new("rpicam-still")
        .args(&[
            "--nopreview",
            "--output", "frame.jpg",
            "-t", "1",
            "--width", "640",
            "--height", "480",
        ])
        .output()?;
    let frame = imgcodecs::imread("frame.jpg", imgcodecs::IMREAD_COLOR)?;
    if frame.empty() {
        println!("Warning: Captured frame is empty!");
    }
    Ok(frame)
}

// Detect persons in the frame using the YOLO model.
// Now takes a confidence threshold as an argument.
fn detect_persons(net: &mut dnn::Net, frame: &Mat, confidence_threshold: f32) -> Result<Vec<Detection>> {
    // Create a blob from the frame.
    let blob = dnn::blob_from_image(
        frame,
        1.0 / 255.0,
        core::Size::new(640, 640),
        core::Scalar::default(),
        true,
        false,
        core::CV_32F,
    )?;
    net.set_input(&blob, "", 1.0, core::Scalar::default())?;

    // Get output layer names and obtain outputs as a vector of Mats.
    let out_blob_names = net.get_unconnected_out_layers_names()?;
    let mut outputs = Vector::<Mat>::new();
    net.forward(&mut outputs, &out_blob_names)?;

    if outputs.len() == 0 {
        println!("Warning: No outputs from net.forward!");
        return Ok(vec![]);
    }

    // Convert each output to a 2D Mat if needed.
    let mut mats = Vector::<Mat>::new();
    for i in 0..outputs.len() {
        let mat = outputs.get(i)?;
        if mat.dims() > 2 {
            let dims = mat.mat_size(); // Vec<i32>
            if dims.is_empty() {
                continue;
            }
            let cols = dims[dims.len() - 1];
            if cols == 0 {
                continue;
            }
            let total = mat.total() as i32;
            let reshaped = mat.reshape(1, total / cols)?;
            mats.push(reshaped.try_clone()?);
        } else {
            mats.push(mat.clone());
        }
    }
    if mats.len() == 0 {
        println!("Warning: No valid output matrices after reshaping.");
        return Ok(vec![]);
    }

    // Optional: check that all mats have the same number of columns and type.
    let first_cols = mats.get(0)?.cols();
    let first_typ = mats.get(0)?.typ();
    for i in 0..mats.len() {
        let mat = mats.get(i)?;
        if mat.cols() != first_cols || mat.typ() != first_typ {
            println!("Warning: Incompatible matrices found in outputs.");
            return Ok(vec![]);
        }
    }

    // Concatenate the output matrices vertically.
    let mut raw_output = Mat::default();
    core::vconcat(&mats, &mut raw_output)?;

    if raw_output.empty() {
        println!("Warning: raw concatenated network output is empty!");
        return Ok(vec![]);
    }

    println!("Raw output dims: {}", raw_output.dims());
    println!("Raw output size: {:?}", raw_output.size()?);

    // Force raw_output into a continuous matrix by copying its data.
    let mut cont_mat = Mat::default();
    raw_output.copy_to(&mut cont_mat)?;

    // If cont_mat has more than 2 dimensions, reshape it into a 2D matrix.
    let output: Mat = if cont_mat.dims() > 2 {
        let dims = cont_mat.mat_size();
        if dims.is_empty() {
            println!("Warning: cont_mat dimensions are empty.");
            return Ok(vec![]);
        }
        let cols = dims[dims.len() - 1];
        if cols == 0 {
            println!("Warning: cont_mat has 0 columns.");
            return Ok(vec![]);
        }
        let total = cont_mat.total() as i32;
        let reshaped = cont_mat.reshape(1, total / cols)?;
        reshaped.try_clone()?
    } else {
        cont_mat.clone()
    };

    println!("Processed output dims: {}", output.dims());
    println!("Processed output size: {:?}", output.size()?);

    let mut detections = Vec::new();
    // Iterate over each row (each detection candidate).
    for row in 0..output.rows() {
        let row_mat = output.row(row)?;
        if row_mat.cols() < 6 {
            println!("Row {} has too few columns: {}", row, row_mat.cols());
            continue;
        }
        // Assume class scores start at column 5.
        let range = core::Range::new(5, output.cols())?;
        let class_scores = row_mat.col_range(&range)?;

        // Confidence is assumed at column 4.
        let confidence = *row_mat.at_2d::<f32>(0, 4)?;
        if confidence > confidence_threshold {
            let mut min_val = 0.0;
            let mut max_val = 0.0;
            let mut min_loc = core::Point::default();
            let mut max_loc = core::Point::default();
            core::min_max_loc(
                &class_scores,
                Some(&mut min_val),
                Some(&mut max_val),
                Some(&mut min_loc),
                Some(&mut max_loc),
                &Mat::default(),
            )?;
            // Assume that the "person" class is at index 0.
            if max_loc.x == 0 && max_val > confidence_threshold.into() {
                let center_x = *row_mat.at_2d::<f32>(0, 0)? * frame.cols() as f32;
                let center_y = *row_mat.at_2d::<f32>(0, 1)? * frame.rows() as f32;
                let width = *row_mat.at_2d::<f32>(0, 2)? * frame.cols() as f32;
                let height = *row_mat.at_2d::<f32>(0, 3)? * frame.rows() as f32;
                // Compute bounding box.
                let x = (center_x - width / 2.0).max(0.0) as i32;
                let y = (center_y - height / 2.0).max(0.0) as i32;
                let mut w = width as i32;
                let mut h = height as i32;
                if x + w > frame.cols() {
                    w = frame.cols() - x;
                }
                if y + h > frame.rows() {
                    h = frame.rows() - y;
                }
                let bbox = core::Rect::new(x, y, w, h);
                detections.push(Detection {
                    centroid: (center_x, center_y),
                    area: width * height,
                    bbox,
                });
            }
        }
    }
    Ok(detections)
}

// Compute the average detection.
fn average_detections(detections: &[Detection]) -> Option<Detection> {
    if detections.is_empty() {
        return None;
    }
    let (sum_x, sum_y, sum_area) = detections.iter().fold((0.0, 0.0, 0.0), |acc, det| {
        (acc.0 + det.centroid.0, acc.1 + det.centroid.1, acc.2 + det.area)
    });
    let count = detections.len() as f32;
    Some(Detection {
        centroid: (sum_x / count, sum_y / count),
        area: sum_area / count,
        bbox: detections[0].bbox,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read YAML configuration file if provided as the first argument.
    let args: Vec<String> = env::args().collect();
    let config: Config = if args.len() > 1 {
        let config_path = &args[1];
        let contents = fs::read_to_string(config_path)?;
        serde_yaml::from_str(&contents)?
    } else {
        // Default configuration.
        Config {
            save_thumbnails: false,
            centroid_threshold: 10.0,
            area_threshold: 500.0,
            confidence_threshold: 0.5,
        }
    };
    println!("Using configuration: {:?}", config);

    // Initialize ROS2.
    let context = Context::new(env::args())?;
    let node = rclrs::create_node(&context, "person_follow_node")?;
    let publisher = node.create_publisher::<Twist>("/cmd_vel", rclrs::QOS_PROFILE_DEFAULT)?;

    // Load the YOLOv5 model.
    let mut net = dnn::read_net_from_onnx("/home/pi/Documents/follow_me/src/yolov5s.onnx")?;
    let mut last_detection: Option<Detection> = None;

    loop {
        let frame = capture_image()?;
        if frame.empty() {
            println!("Empty frame, skipping processing.");
            sleep(Duration::from_secs(1));
            continue;
        }
        let detections = detect_persons(&mut net, &frame, config.confidence_threshold)?;
        
        // Optionally save thumbnails.
        if config.save_thumbnails {
            fs::create_dir_all("/home/pi/Documents/follow_me/thumbnails/")?;
            for (i, det) in detections.iter().enumerate() {
                let roi = Mat::roi(&frame, det.bbox)?;
                let mut thumb = roi.try_clone()?;
                imgproc::rectangle(
                    &mut thumb,
                    det.bbox,
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0,
                )?;
                imgproc::put_text(
                    &mut thumb,
                    "person",
                    core::Point::new(5, 20),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                    1,
                    imgproc::LINE_8,
                    false,
                )?;
                let filename = format!(
                    "/home/pi/Documents/follow_me/thumbnails/thumbnail_{}_{}.jpg",
                    Utc::now().timestamp(),
                    i
                );
                imgcodecs::imwrite(&filename, &thumb, &Vector::new())?;
            }
        }
        
        let avg_detection = average_detections(&detections);

        // Compare current detection with previous detection and compute a Twist message.
        if let (Some(last), Some(current)) = (&last_detection, &avg_detection) {
            let mut twist = Twist::default();
            if current.centroid.0 > last.centroid.0 + config.centroid_threshold {
                twist.angular.z = -0.2;
            } else if current.centroid.0 < last.centroid.0 - config.centroid_threshold {
                twist.angular.z = 0.2;
            }
            if current.area > last.area + config.area_threshold {
                twist.linear.x = -0.1;
            } else if current.area < last.area - config.area_threshold {
                twist.linear.x = 0.1;
            }
            println!(
                "Detected persons. Sending Twist: linear.x = {}, angular.z = {}",
                twist.linear.x, twist.angular.z
            );
            publisher.publish(&twist)?;
        }

        last_detection = avg_detection;
        sleep(Duration::from_secs(1));
    }
}
