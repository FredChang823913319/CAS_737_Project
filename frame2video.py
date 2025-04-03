# merge_frames_to_video.py

import cv2
import os
import sys
import numpy as np

def merge_frames_to_video(frame_folder, output_video_path, fps=30, video_size=None):
    # If the output video path is just a folder, append a default filename
    if os.path.isdir(output_video_path):
        output_video_path = os.path.join(output_video_path, 'output_video.mp4')
    
    # Ensure output video path has the correct file extension
    if not output_video_path.endswith('.mp4'):
        print("Error: Output video path must have a .mp4 extension.")
        return
    
    # Ensure the output folder exists
    output_folder = os.path.dirname(output_video_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of all image files in the frame folder
    frame_files = [f for f in os.listdir(frame_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort the frames by the numeric part in the filename (assuming the pattern "frame_0000.jpg")
    frame_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # print(frame_files)

    # Check if there are frames to process
    if not frame_files:
        print("No frames found in the folder.")
        return
    
    # Read the first frame to get the video size (width, height)
    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    
    # for frame_file in frame_files:
    #   frame = cv2.imread(os.path.join(frame_folder, frame_file))
    #   if frame is not None:
    #     print(f"Frame {frame_file} size: {frame.shape[1]}x{frame.shape[0]}")  # Width x Height

    if first_frame is None:
        print("Unable to read the first frame.")
        return
    
    if not video_size:
        video_size = (first_frame.shape[1], first_frame.shape[0])

    # Create VideoWriter object to write frames into a video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, video_size)

    # Iterate over each frame and write it to the video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frame_folder, frame_file))
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Unable to read frame {frame_file}. Skipping.")

    # Release the VideoWriter and finish
    out.release()
    print(f"Video saved at {output_video_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python merge_frames_to_video.py <frame_folder> <output_video_path> [fps]")
    else:
        frame_folder = sys.argv[1]
        output_video_path = sys.argv[2]
        
        # Check if fps is provided, else use default 30
        if len(sys.argv) > 3:
            fps = int(sys.argv[3])
        else:
            fps = 30  # Default value, can be changed to 1 or other value if desired. 

        merge_frames_to_video(frame_folder, output_video_path, fps)
