import cv2
import matplotlib.pyplot as plt
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

def save_normalized_points(output_dir, image_path, height, width):
    """Saves the normalized points to a file."""
    global clicked_points
    # Normalize coordinates
    normalized_points = [(y / height, x / width) for x, y in clicked_points]
    
    # Generate output file name
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # output_file = os.path.join(output_dir, f"{base_name}_gt.txt")
    output_file = os.path.join(output_dir, f"{base_name}.txt")
    
    # Save points to file
    with open(output_file, "w") as file:
        for x, y in normalized_points:
            file.write(f"{x},{y}\n")
    
    print(f"Normalized Keypoints saved to {output_file}")

def get_normalized_points(image_path, output_dir):
    """Loads an image, allows user to select keypoints, and normalizes coordinates."""
    global clicked_points, current_joint_index
    clicked_points = []
    current_joint_index = 0
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Show image
    print(f"Select: {joints[current_joint_index]}")
    cv2.imshow("Select Joints", img)
    cv2.setMouseCallback("Select Joints", click_event)
    
    while current_joint_index < len(joints):
        cv2.waitKey(1)  # Keep checking until all joints are selected
    
    cv2.destroyAllWindows()
    save_normalized_points(output_dir, image_path, height, width)

    # return {joint: coord for joint, coord in zip(joints, normalized_points)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python script.py <image_directory> <output_dir>")
        sys.exit(1)
    
    image_directory = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process all images in the directory
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            image_path = os.path.join(image_directory, filename)
            get_normalized_points(image_path, output_dir)
