#!/usr/bin/env python3
import cv2
import mediapipe as mp
import os
import argparse
import ssl

def write_landmarks(pose_landmarks, output_file_path):
    # # Define the corresponding joint names for each landmark index (33 landmarks)
    # joint_names = [
    #     'Nose', 'Left Eye Inner', 'Left Eye', 'Left Eye Outer', 'Right Eye Inner', 'Right Eye', 'Right Eye Outer',
    #     'Left Ear', 'Right Ear', 'Mouth Left', 'Mouth Right', 'Left Shoulder', 'Right Shoulder', 'Left Elbow',
    #     'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee',
    #     'Left Ankle', 'Right Ankle', 'Left Foot Index', 'Right Foot Index',
    #     'Left Shoulder (Upper)', 'Right Shoulder (Upper)', 'Left Hip (Upper)', 'Right Hip (Upper)',
    #     'Left Knee (Upper)', 'Right Knee (Upper)', 'Left Ankle (Upper)', 'Right Ankle (Upper)'
    # ]
    mediapipe_to_movenet = {
        0: "Nose", 2: "Left Eye", 5: "Right Eye", 7: "Left Ear", 8: "Right Ear",
        11: "Left Shoulder", 12: "Right Shoulder", 13: "Left Elbow", 14: "Right Elbow",
        15: "Left Wrist", 16: "Right Wrist", 23: "Left Hip", 24: "Right Hip",
        25: "Left Knee", 26: "Right Knee", 27: "Left Ankle", 28: "Right Ankle"
    }

    # Prepare the output text for the landmarks coordinates
    # output_text = ""
    # for i, landmark in enumerate(pose_landmarks):
    #     # print(f"{i} - (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
    #     # if i < len(joint_names):
    #     output_text += f"{joint_names[i]} - (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})\n"
    # Prepare the output text for the selected landmarks
    output_text = ""
    for index, name in mediapipe_to_movenet.items():
        landmark = pose_landmarks[index]
        # output_text += f"{name} - (x: {landmark.x:.4f}, y: {landmark.y:.4f}, z: {landmark.z:.4f})\n"
        output_text += f"{landmark.y:.4f},{landmark.x:.4f}\n"  # Store (y, x) only


    # Save the coordinates to a text file
    with open(output_file_path, 'w') as file:
        file.write(output_text)
    return output_file_path



def mediapipe_eval(input_folder, output_folder, output_landmark_folder):
    # Create output folder if it doesn't exist
    # Disable SSL verification (not good for production code, security risks)
    ssl._create_default_https_context = ssl._create_unverified_context
    os.makedirs(output_folder, exist_ok=True)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

    # Process each frame in the input folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Convert to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # Draw annotations
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # for landmark in results.pose_landmarks.landmark:
            #     print(f"Landmark {landmark} coordinates: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
            
            output_landmark_filepath = output_landmark_folder + filename.split(".")[0] + ".txt"
            write_landmarks(results.pose_landmarks.landmark, output_landmark_filepath)
                
            # Save the annotated image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)

            # break

    # Release resources
    pose.close()
    print(f"Annotation complete. Check the '{output_folder}' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate frames with MediaPipe Pose.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing frames.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for annotated frames.")
    parser.add_argument("output_landmark_folder", type=str, help="Path to the output folder for landmarks per frame.")
    args = parser.parse_args()
    
    mediapipe_eval(args.input_folder, args.output_folder, args.output_landmark_folder)
