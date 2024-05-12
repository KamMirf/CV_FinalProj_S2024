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
    all_files_results = []
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

            # Create a dictionary for the current file with class names as keys and counts as values
            if file_counts:
                results_dict = {class_names[cls_id]: count for cls_id, count in file_counts.items()}
                all_files_results.append((label_file, results_dict))

    return all_files_results

def main():
    labels_dir = "../yolo_outputs/test_run2/labels"  # Specify the directory containing label files
    confidence_threshold = 0.5  # Set the confidence threshold for filtering detections
    results = process_label_files(labels_dir, confidence_threshold)

    # Print all results
    for file_name, detections in results:
        print(f"Results for {file_name}:")
        for class_name, count in detections.items():
            print(f"  {class_name}: {count}")
        print()  # Add a newline for better separation between files

if __name__ == "__main__":
    main()
