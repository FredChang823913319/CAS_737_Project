import cv2
import os

# Joint names mapping
joints = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee", "Right Knee",
    "Left Ankle", "Right Ankle"
]

# Global variables to store click coordinates
clicked_points = []
current_joint_index = 0

def click_event(event, x, y, flags, param):
    """Callback function to record click coordinates."""
    global current_joint_index
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"{joints[current_joint_index]} selected at: {x}, {y}")
        current_joint_index += 1
        if current_joint_index >= len(joints):
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Ensure window properly closes
        else:
            print(f"Select: {joints[current_joint_index]}")

def save_normalized_points(landmark_dir, image_path, height, width):  
    """Saves the normalized points to a file."""
    global clicked_points
    normalized_points = [(y / height, x / width) for x, y in clicked_points]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(landmark_dir, f"{base_name}.txt")  
    with open(output_file, "w") as file:
        for x, y in normalized_points:
            file.write(f"{x},{y}\n")
    print(f"Normalized Keypoints saved to {output_file}")

def save_annotated_image(image_path, image_save_dir):  
    """Saves the image with selected keypoints marked."""
    img = cv2.imread(image_path)
    for x, y in clicked_points:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Draw red dot for each keypoint
    base_name = os.path.basename(image_path)
    annotated_path = os.path.join(image_save_dir, base_name)  
    cv2.imwrite(annotated_path, img)
    print(f"Annotated image saved to {annotated_path}")

def get_normalized_points(image_path, landmark_dir, image_save_dir):  
    """Loads an image, allows user to select keypoints, and normalizes coordinates."""
    global clicked_points, current_joint_index
    clicked_points = []
    current_joint_index = 0
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    print(f"Select: {joints[current_joint_index]}")
    cv2.imshow("Select Joints", img)
    cv2.setMouseCallback("Select Joints", click_event)
    
    while current_joint_index < len(joints):
        cv2.waitKey(1)  # Keep checking until all joints are selected
    
    cv2.destroyAllWindows()
    save_normalized_points(landmark_dir, image_path, height, width)  
    save_annotated_image(image_path, image_save_dir)  

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python script.py <image_directory> <landmark_dir> <image_save_dir>")
        sys.exit(1)
    
    image_directory = sys.argv[1]
    landmark_dir = sys.argv[2]  
    image_save_dir = sys.argv[3]  
    
    if not os.path.exists(landmark_dir):  
        os.makedirs(landmark_dir)
    if not os.path.exists(image_save_dir):  
        os.makedirs(image_save_dir)
    
    # Process all images in the directory
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            image_path = os.path.join(image_directory, filename)
            get_normalized_points(image_path, landmark_dir, image_save_dir)  
