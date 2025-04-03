#!/usr/bin/env python3
import cv2
import os
import argparse
import numpy as np
import tensorflow as tf
import numpy as np
from movenet_util import draw_prediction_on_image

# Function to remove white borders
def remove_white_borders(image, threshold=240):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find all non-white pixels
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Find contours of non-white regions
    coords = cv2.findNonZero(thresh)  # Get all non-zero points
    x, y, w, h = cv2.boundingRect(coords)  # Get bounding box

    # Crop the image using bounding box
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

# Define the function for writing landmarks to a file
def write_landmarks(keypoints, output_file_path):
    joint_names = [
        'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
        'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', 
        'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 
        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
    ]

    # Extract keypoints for landmarks (keypoints is of shape [1, 1, 17, 3])
    keypoints = keypoints[0][0]  # This gives the shape (17, 3), i.e., 17 landmarks with (x, y, score)

    output_text = ""
    for i, landmark in enumerate(keypoints):
        # x, y, score = landmark
        # output_text += f"{joint_names[i]} - (x: {x}, y: {y}, score: {score})\n"
        x, y, _ = landmark  # Unpacking x, y, and score, but ignoring score
        output_text += f"{x},{y}\n"  # Storing x and y only, separated by a comma

    # Save the coordinates to a text file
    with open(output_file_path, 'w') as file:
        file.write(output_text)
    return output_file_path

# Function to load the TensorFlow Lite model
def load_movenet_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_movenet_tflite(input_image, interpreter):
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

# Evaluating the pose estimation using movenet
def movenet_eval(input_folder, output_folder, output_landmark_folder):
    # Load the MoveNet model
    model_path = 'movenet_lite/model.tflite'  # movenet_lightning_f16
    interpreter = load_movenet_tflite_model(model_path)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each frame in the input folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = tf.io.read_file(image_path)

            if image is None:
              continue
            image = tf.image.decode_jpeg(image)
            input_image = tf.expand_dims(image, axis=0)
            input_image = tf.image.resize_with_pad(input_image, 192, 192)    # movenet_lightning_f16 input_size = 192

            # Run model inference.
            keypoints_with_scores = run_movenet_tflite(input_image, interpreter)
            print(keypoints_with_scores)
            print(keypoints_with_scores.shape)
            # break
            # Visualize the predictions with image.
            display_image = tf.expand_dims(image, axis=0)
            display_image = tf.cast(tf.image.resize_with_pad(
                display_image, 1280, 1280), dtype=tf.int32)
            output_overlay = draw_prediction_on_image(
                np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)


            output_path = os.path.join(output_folder, filename)
            # output_bgr = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(output_path, output_bgr)

            # Convert overlay from RGB to BGR (if needed)
            output_bgr = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
            # Remove white borders
            output_bgr_cropped = remove_white_borders(output_bgr)
            # Save the cropped image
            cv2.imwrite(output_path, output_bgr_cropped)

            # Prepare the landmark file path
            output_landmark_filepath = os.path.join(output_landmark_folder, filename.split(".")[0] + ".txt")
            write_landmarks(keypoints_with_scores, output_landmark_filepath)

            # break 

    print(f"Annotation complete. Check the '{output_folder}' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate frames with MoveNet Pose.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing frames.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for annotated frames.")
    parser.add_argument("output_landmark_folder", type=str, help="Path to the output folder for landmarks per frame.")
    args = parser.parse_args()

    movenet_eval(args.input_folder, args.output_folder, args.output_landmark_folder)
