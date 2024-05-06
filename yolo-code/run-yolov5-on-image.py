import os
import subprocess
import sys
from collections import Counter
import datetime

# List of class names from data.yaml
class_names = [
    "apple", "banana", "beef", "blueberries", "bread", "butter",
    "carrot", "cheese", "chicken", "chicken_breast", "chocolate",
    "corn", "eggs", "flour", "goat_cheese", "green_beans",
    "ground_beef", "ham", "heavy_cream", "lime", "milk",
    "mushrooms", "onion", "potato", "shrimp", "spinach",
    "strawberries", "sugar", "sweet_potato", "tomato"
]

import os
import subprocess

def detect_image(input_image_path):
    base_dir = os.path.dirname(__file__)
    yolov5_dir = os.path.join(base_dir, '../yolov5')
    detect_script = os.path.join(yolov5_dir, 'detect.py')
    weights_path = os.path.join(base_dir, '../weights/best.pt')
    
    # Create a unique directory name based on the current timestamp
    unique_dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f'../yolo_outputs/{unique_dir_name}')

    os.makedirs(output_dir, exist_ok=True)

    command = [
        'python', detect_script,
        '--weights', weights_path,
        '--source', input_image_path,
        '--conf-thres', '0.60',
        '--iou-thres', '0.45',
        '--project', output_dir,
        '--exist-ok',
        '--save-txt',
        '--save-conf'
    ]
    subprocess.run(command, check=True)
    return os.path.join(output_dir, 'exp/labels')


def process_label_files(labels_dir):
    all_files_results = []
    # Process each label file
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            file_counts = Counter()
            with open(os.path.join(labels_dir, label_file), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 6 and float(parts[5]) >= 0.5:  # Confidence threshold
                        cls_id = int(parts[0])
                        file_counts[cls_id] += 1

            # Create a dictionary for the current file with class names as keys and counts as values
            if file_counts:
                results_dict = {class_names[cls_id]: count for cls_id, count in file_counts.items()}
                all_files_results.append((label_file, results_dict))

    return all_files_results

def main(input_image_path):
    labels_dir = detect_image(input_image_path)
    results = process_label_files(labels_dir)

    # Print all results
    for file_name, detections in results:
        print(f"Results for {file_name}:")
        for class_name, count in detections.items():
            print(f"  {class_name}: {count}")
        print()  # Add a newline for better separation between files

if __name__ == "__main__":
    """

    python run-yolov5-on-image.py path/to/your/image.jpg

    working exmaple:
    python3 run-yolov5-on-image.py ../images-to-test-yolov5-on/1.jpeg
    
    """
    main(sys.argv[1])

