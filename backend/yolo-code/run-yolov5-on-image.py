import os
import subprocess
import sys
from collections import Counter
import datetime
from PIL import Image
import json

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
    yolov5_dir = os.path.join(base_dir, 'yolov5')
    detect_script = os.path.join(yolov5_dir, 'detect.py')
    weights_path = os.path.join(base_dir, 'weights/best.pt')
    
    # Create a unique directory name based on the current timestamp
    unique_dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f'yolo_outputs/{unique_dir_name}')

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
    return os.path.join(output_dir, 'exp/labels'), input_image_path

def process_label_files(labels_dir, img_width, img_height):
    results = {}
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        cls_id = int(parts[0])
                        # Ensure cls_id is within the range of class_names
                        if cls_id < len(class_names):
                            x_center = float(parts[1]) * img_width
                            y_center = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            confidence = float(parts[5])

                            box = [x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2]

                            class_name = class_names[cls_id]  # This should now be safe
                            if class_name not in results:
                                results[class_name] = {"count": 0, "boxes": [], "confidences": []}
                            results[class_name]["count"] += 1
                            results[class_name]["boxes"].append(box)
                            results[class_name]["confidences"].append(confidence)
                        else:
                            print(f"Warning: Class ID {cls_id} out of range.")
    return results


def main(input_image_path):
    labels_dir, image_path = detect_image(input_image_path)
    
    # Open the image and get dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    results = process_label_files(labels_dir, img_width, img_height)

    # Print results as JSON
    print(json.dumps(results))


    # Print results in a formatted way
    #print("Detailed Results:")
    #for class_name, info in results.items():
    #    print(f"{class_name}: {info['count']}")
    #    for box, conf in zip(info['boxes'], info['confidences']):
    #        print(f"  Box: {box}, Confidence: {conf}")
    #print()  # Add a newline for better separation

if __name__ == "__main__":
    """

    python run-yolov5-on-image.py path/to/your/image.jpg

    working exmaple:
    python3 run-yolov5-on-image.py images-to-test-yolov5-on/1.jpeg
    
    """
    if len(sys.argv) != 2:
        print("Usage: python run-yolov5-on-image.py <path/to/your/image.jpg>")
        sys.exit(1)
    main(sys.argv[1])

