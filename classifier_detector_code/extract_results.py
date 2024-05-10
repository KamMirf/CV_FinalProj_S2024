from collections import Counter
from classifier_to_detector import CustomModel, VGGModel
import tensorflow as tf
import hyperparameters as hp
import cv2
import matplotlib.pyplot as plt
import sys



# List of class names from data.yaml
class_names = [
    "apple", "banana", "beef", "blueberries", "bread", "butter",
    "carrot", "cheese", "chicken", "chicken_breast", "chocolate",
    "corn", "eggs", "flour", "goat_cheese", "green_beans",
    "ground_beef", "ham", "heavy_cream", "lime", "milk",
    "mushrooms", "onion", "potato", "shrimp", "spinach",
    "strawberries", "sugar", "sweet_potato", "tomato"
]



def process_image(final_boxes, confidences, classIDs):
    # Initialize results dictionary to only include class names that appear in classIDs
    unique_classIDs = set(classIDs)  # Remove duplicates to optimize initialization
    results_dict = {class_names[classID]: {'count': 0, 'boxes': [], 'confidences': []} for classID in unique_classIDs}
    
    # Iterate over each detected object
    for box, confidence, classID in zip(final_boxes, confidences, classIDs):
        class_name = class_names[classID]
        # Ensure the class_name is in the dictionary (it should always be unless class_names or classIDs is inconsistent)
        if class_name in results_dict:
            results_dict[class_name]['count'] += 1
            results_dict[class_name]['boxes'].append(box)
            results_dict[class_name]['confidences'].append(confidence)
    
    return results_dict

def detect_image(input_image_path, model_type="Custom"):
    custom_checkpoint = 'checkpoints/Custom/050724-130938/custom.e010-acc0.8951.weights.h5'
    vgg_head_checkpoint = 'checkpoints/vgg_model/050324-233826/vgg.e001-acc0.7443.weights.h5'
    vgg_body_weight_path = '../vgg16_imagenet.h5'

    if model_type == "VGG":
        model = VGGModel()
        model(tf.keras.Input(shape=(224, 224, 3)))
        model.head.load_weights(vgg_head_checkpoint)
        model.vgg16.load_weights(vgg_body_weight_path, by_name=True)
    else:  # Default to Custom
        model = CustomModel()
        model(tf.keras.Input(shape=(hp.window_size, hp.window_size, 3)))
        model.load_weights(custom_checkpoint)

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    final_boxes, confidences, classIDs = run_detection_and_visualization(input_image_path, model, class_names, model_type)

    results = process_image(final_boxes, confidences, classIDs)

    # Print results in a formatted way
    print("##########################################")
    print("RESULTS: ")
    print(results)
    print("##########################################")
    
    print("Detailed Results:")
    for class_name, info in results.items():
        print(f"{class_name}: {info['count']}")
        for box, conf in zip(info['boxes'], info['confidences']):
            print(f"  Box: {box}, Confidence: {conf:.2f}")
    print()  # Add a newline for better separation

    return results

# different scaled images
def image_pyramid(image, scale=1.5, min_size=(224, 224)): 
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = tf.image.resize(image, (w, w))
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image

# sliding window across image
def sliding_window(image, step_size, window_size):

    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            window = (image[y:y + window_size[1], x:x + window_size[0]])
            yield (x, y, window)


def object_detection(image, model, scale=1.5, win_size=(224, 224), step_size=32, threshold=0.5, max_box_area=0.25):
    boxes = []
    confidences = []
    classIDs = []

    # Maximum area of the box as a fraction of the image area
    max_area_px = image.shape[0] * image.shape[1] * max_box_area

    # Create an image pyramid and sliding a window
    for resized in image_pyramid(image, scale, min_size=win_size):
        for (x, y, window) in sliding_window(resized, step_size=step_size, window_size=win_size):

            if x <= 30 or x >= image.shape[0] - 30 or y <= 30 or y >= image.shape[1] - 30:
                continue
            window = tf.expand_dims(window, axis=0)
            window = tf.image.resize(window, win_size)
            preds = model.predict(window)
            classID = tf.argmax(preds[0])
            confidence = preds[0][classID]

            if confidence > threshold:
                scale_x = image.shape[1] / resized.shape[1]
                scale_y = image.shape[0] / resized.shape[0]
                box = [x * scale_x, y * scale_y, (x + win_size[0]) * scale_x, (y + win_size[1]) * scale_y]
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                box_area = box_width * box_height

                if box_area <= max_area_px:
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID.numpy())

    if boxes:
        boxes = tf.constant(boxes, dtype=tf.float32)  # Ensure boxes are a 2D Tensor
        confidences = tf.constant(confidences, dtype=tf.float32)

        idxs = tf.image.non_max_suppression(boxes, confidences, max_output_size=50, iou_threshold=0.5)
        final_boxes = [boxes[i].numpy().tolist() for i in idxs]  # Convert tensor indices to list of boxes
    else:
        final_boxes = []

    return final_boxes, confidences.numpy().tolist(), classIDs


def draw_boxes(image, boxes, confidences, classIDs, classes):
    for (box, score, classID) in zip(boxes, confidences, classIDs):
        (startX, startY, endX, endY) = box
        startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
        label = f"{classes[classID]}: {score:.2f}"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def run_detection_and_visualization(image_path, model, classes, model_type):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    wind_sz = (224, 224) if model_type == "VGG" else (hp.window_size, hp.window_size)

    # Perform object detection
    final_boxes, confidences, classIDs = object_detection(image, model, scale=1.5, win_size=wind_sz, step_size=64, threshold=0.99, max_box_area=0.08)

    # Draw bounding boxes on the image
    output_image = draw_boxes(image.copy(), final_boxes, confidences, classIDs, classes)

    # Display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()
    return final_boxes, confidences, classIDs
   

#detect_image('data/test/images/DSC_5941_JPG_jpg.rf.7f34ef03affd2f952f6519e8506d8cdc.jpg', "Custom")

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_result.py /path/to/image/file")
        return

    input_image_path = sys.argv[1]
    # Explicitly set the model_type to "Custom" when calling the detect_image function
    detect_image(input_image_path, model_type="Custom")

if __name__ == "__main__":
    """
    python3 extract_results.py data/test/images/DSC_5941_JPG_jpg.rf.7f34ef03affd2f952f6519e8506d8cdc.jpg
    """
    main()