import cv2
import os
import argparse

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames
        frame = cv2.flip(frame, 0)  # Flip the frame vertically
        frame = cv2.flip(frame, 1)  # Flip horizontally (left to right)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate frame 90 degrees counterclockwise
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        
    
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("output_folder", type=str, help="Folder to save extracted frames")
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output_folder)