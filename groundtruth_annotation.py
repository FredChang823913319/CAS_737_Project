import os
import torch
import numpy as np
import tensorflow as tf
import cv2
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

def read_and_convert_to_tensor(data_directory, image_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    tensors = {}
    
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        imagename = os.path.splitext(filename)[0] + ".jpg"  # Assuming image format is .jpg
        imagepath = os.path.join(image_directory, imagename)
        output_imagepath = os.path.join(output_directory, imagename)
        
        if os.path.isfile(filepath) and os.path.isfile(imagepath):
            data = np.loadtxt(filepath, delimiter=',')  # Load the file as a NumPy array
            
            if data.shape[0] != 17 or data.shape[1] != 2:
                raise ValueError(f"File {filename} does not have the expected shape (17,2)")
            
            data = np.hstack((data, np.ones((17, 1))))  # Append a 0 as the third value
            tensor = torch.tensor(data, dtype=torch.float32).reshape(1, 1, 17, 3)
            
            image = tf.io.read_file(imagepath)
            if image is None:
                continue
            image = tf.image.decode_jpeg(image)
            # Visualize the predictions with image.
            display_image = tf.expand_dims(image, axis=0)
            display_image = tf.cast(tf.image.resize_with_pad(
                display_image, 1280, 1280), dtype=tf.int32)
            output_overlay = draw_prediction_on_image(
                np.squeeze(display_image.numpy(), axis=0), tensor)

            # Convert overlay from RGB to BGR (if needed)
            output_bgr = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
            # Remove white borders
            # output_bgr_cropped = remove_white_borders(output_bgr)
            # Save the cropped image
            cv2.imwrite(output_imagepath, output_bgr)
            # cv2.imwrite(output_imagepath, output_bgr_cropped)
            
            tensors[filename] = (tensor, image)
    
    return tensors

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert files in a directory to tensors and match them with images.")
    parser.add_argument("data_directory", type=str, help="Path to the directory containing the data files.")
    parser.add_argument("image_directory", type=str, help="Path to the directory containing the images.")
    parser.add_argument("output_directory", type=str, help="Path to the directory to save output images.")
    args = parser.parse_args()
    
    tensors = read_and_convert_to_tensor(args.data_directory, args.image_directory, args.output_directory)
    
    if tensors:
        first_tensor, first_image = next(iter(tensors.values()))
        print(first_tensor.shape)  # Expected output: torch.Size([1, 1, 17, 3])
