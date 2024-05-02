import os
from collections import Counter

# List of class names from data.yaml
class_names = [
    "apple", "banana", "beef", "blueberries", "bread", "butter",
    "carrot", "cheese", "chicken", "chicken_breast", "chocolate",
    "corn", "eggs", "flour", "goat_cheese", "green_beans",
    "ground_beef", "ham", "heavy_cream", "lime", "milk",
    "mushrooms", "onion", "potato", "shrimp", "spinach",
    "strawberries", "sugar", "sweet_potato", "tomato"
]

def process_label_files(labels_dir, confidence_threshold):
    """Processes each label file in the directory, printing class counts for detections above a confidence threshold."""
    # Process each label file
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            file_counts = Counter()
            with open(os.path.join(labels_dir, label_file), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 6 and float(parts[5]) >= confidence_threshold:
                        cls_id = int(parts[0])
                        file_counts[cls_id] += 1

            # Print the count of each class for the current file
            if file_counts:
                print(f"Results for {label_file}:")
                for cls_id, count in file_counts.items():
                    print(f"  {count} {class_names[cls_id]}{'s' if count > 1 else ''}")
                print()  # Add a newline for better separation between files

def main():
    labels_dir = "../yolo_outputs/test_run3/labels"  # Specify the directory containing label files
    confidence_threshold = 0.5  # Set the confidence threshold for filtering detections
    process_label_files(labels_dir, confidence_threshold)

if __name__ == "__main__":
    main()
