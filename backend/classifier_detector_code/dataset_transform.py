import os
from PIL import Image

# List of classes
classes = [
    "apple", "banana", "beef", "blueberries", "bread",
    "butter", "carrot", "cheese", "chicken", "chicken_breast",
    "chocolate", "corn", "eggs", "flour", "goat_cheese",
    "green_beans", "ground_beef", "ham", "heavy_cream", "lime",
    "milk", "mushrooms", "onion", "potato", "shrimp", "spinach",
    "strawberries", "sugar", "sweet_potato", "tomato"
]

def split_data(data_dir, output_dir, testing):
    """
    Split data into class folders based on class index.
    
    Arguments:
        data_dir (str): Directory containing images and label files.
        output_dir (str): Directory to save the split data.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a folder for each class
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    # Loop through each image in the data directory
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.endswith(".jpg"):
                image_path = os.path.join(root, name)
                
                # Load the image
                img = Image.open(image_path)
                img_width, img_height = img.size
                
                # Read the corresponding label file
                if (testing):
                    label_path = image_path.replace("test/images", "test/labels").replace(".jpg", ".txt")
                else:
                    label_path = image_path.replace("train/images", "train/labels").replace(".jpg", ".txt")
                with open(label_path, "r") as f:
                    lines = f.readlines()
                
                # Loop through each bounding box in the label file
                for line in lines:
                    parts = line.strip().split()
                    class_idx = int(parts[0])  # Extract class index
                    
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert relative coordinates to absolute coordinates
                    x_min = int((x_center - width / 2) * img_width)
                    y_min = int((y_center - height / 2) * img_height)
                    x_max = int((x_center + width / 2) * img_width)
                    y_max = int((y_center + height / 2) * img_height)
                    
                    # Crop the image based on the bounding box
                    cropped_img = img.crop((x_min, y_min, x_max, y_max))
                    
                    # Save the cropped image to the corresponding class folder
                    class_name = classes[class_idx]
                    output_image_filename = f"{name.split('.')[0]}_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
                    output_image_path = os.path.join(output_dir, class_name, output_image_filename)
                    cropped_img.save(output_image_path)

#to get cropped test data
data_dir = "data/test"
output_dir = "cropped_data/test"
split_data(data_dir, output_dir, True)

#to get cropped train data
data_dir = "data/train"
output_dir = "cropped_data/train"
split_data(data_dir, output_dir, False)
