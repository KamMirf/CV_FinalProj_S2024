import os
from collections import Counter

# List of class names from the data.yaml
class_names = [
    "apple", 
    "banana", 
    "beef", 
    "blueberries", 
    "bread", 
    "butter",
    "carrot", 
    "cheese", 
    "chicken", 
    "chicken_breast", 
    "chocolate",
    "corn", 
    "eggs", 
    "flour", 
    "goat_cheese", 
    "green_beans",
    "ground_beef", 
    "ham", 
    "heavy_cream", 
    "lime", 
    "milk",
    "mushrooms",
    "onion", 
    "potato", 
    "shrimp", 
    "spinach",
    "strawberries", 
    "sugar", 
    "sweet_potato",
    "tomato"
]

# Directory containing label files
labels_dir = "../yolo_outputs/test_run3/labels"

# Process each label file
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        # Counter for detected classes in the current file
        file_counts = Counter()

        with open(os.path.join(labels_dir, label_file), 'r') as file:
            detections = [line.split()[0] for line in file.readlines()]
            file_counts.update(detections)

        # Print the count of each class for the current file
        print(f"Results for {label_file}:")
        for cls_id, count in file_counts.items():
            print(f"  {count} {class_names[int(cls_id)]}s")
        print()  # Add a newline for better separation between files