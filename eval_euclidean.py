import os
import numpy as np
import statistics   

def read_points(file_path):
    """Read (x, y) points from a file and skip invalid lines."""
    points = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    points.append(tuple(map(float, line.strip().split(','))))
                except ValueError:
                    print(f"Skipping invalid line in {file_path}: {line.strip()}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return points

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def compare_directories(dir1, dir2):
    """Compare files with the same name in two directories and compute the total Euclidean distance."""
    files1 = set(f for f in os.listdir(dir1) if f.endswith(".txt"))
    files2 = set(f for f in os.listdir(dir2) if f.endswith(".txt"))
    print(f"sorted files1: {sorted(files1)}")
    print(f"sorted files2: {sorted(files2)}")

    common_files = files1 & files2  # Get only matching filenames
    print(f"Matching files: {sorted(common_files)}")

    distances = {}

    for filename in sorted(common_files):
        path1 = os.path.join(dir1, filename)
        path2 = os.path.join(dir2, filename)

        points1 = read_points(path1)
        points2 = read_points(path2)

        if len(points1) != len(points2):
            print(f"Skipping {filename}: Different number of points ({len(points1)} vs {len(points2)}).")
            continue

        total_distance = sum(euclidean_distance(p1, p2) for p1, p2 in zip(points1, points2))
        distances[filename] = total_distance
    
    
    if distances:                       
        # variance (divide by N) 
        overall_var = statistics.pvariance(distances.values())
        distances['distance sum'] = sum(distances.values())
        distances['variance'] = overall_var
    else:
        distances['variance'] = 0.0   
        distances['distance sum'] = 0.0       
    return distances

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python script.py <directory1> <directory2>")
        sys.exit(1)

    dir1, dir2 = sys.argv[1], sys.argv[2]
    result = compare_directories(dir1, dir2)

    print("\nEuclidean distances:")
    for file, distance in result.items():
        print(f"{file}: {distance:.4f}")
