import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        self.num_classes = 30
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        for layer in self.vgg16:
            layer.trainable = False


        self.head = [
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.num_classes, activation='softmax') # 30 classes

        ]

        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """


        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy() # 30 classes

        return loss_obj(labels, predictions)
    

################################ DETECTOR CODE #########################################


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
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


# object detection in an image      
def object_detection(image, model, scale=1.5, win_size=(224, 224), step_size=32, threshold=0.5):
    boxes = []
    confidences = []
    classIDs = []

    # Create an image pyramid and sliding a window
    for resized in image_pyramid(image, scale, min_size=win_size):
        for (x, y, window) in sliding_window(resized, step_size=step_size, window_size=win_size):
            if window.shape[0] != win_size[1] or window.shape[1] != win_size[0]:
                continue
            # Preprocess the window for classification
            window = tf.expand_dims(window, axis=0)
            window = tf.image.resize(window, (224, 224))
            preds = model(window, training=False)
            classID = tf.argmax(preds[0])
            confidence = preds[0][classID]

            # Filter out weak predictions
            if confidence > threshold:
                box = (x, y, x + win_size[0], y + win_size[1])
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID.numpy())

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = tf.image.non_max_suppression(boxes, confidences, max_output_size=50, iou_threshold=0.5)
    final_boxes = [boxes[i] for i in idxs]

    return final_boxes, confidences, classIDs

# bounding boxes on image 
def draw_boxes(image, boxes, confidences, classIDs, classes):
    for (box, score, classID) in zip(boxes, confidences, classIDs):
        (startX, startY, endX, endY) = box
        label = f"{classes[classID]}: {score:.2f}"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def run_detection_and_visualization(image_path, model, classes):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform object detection
    final_boxes, confidences, classIDs = object_detection(image, model)

    # Draw bounding boxes on the image
    output_image = draw_boxes(image.copy(), final_boxes, confidences, classIDs, classes)

    # Display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()

###########################################################################################

def load_data(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Validation data not to be augmented
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse') 

    # Flow validation images in batches using val_datagen generator
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse')

    return train_generator, validation_generator

def compile_model(model):
    model.compile(optimizer=model.optimizer,
                  loss=model.loss_fn,
                  metrics=['accuracy'])
    
def train_model(model, train_data, val_data, epochs=10):
    history = model.fit(
        train_data,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        validation_data=val_data,
        validation_steps=val_data.samples // val_data.batch_size,
        epochs=epochs,
        verbose=2)
    return history

def evaluate_model(model, test_data):
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test Accuracy: {test_acc}")
    print(f"Test Loss: {test_loss}")
    
def run():
    train_dir = "need data"
    val_dir = "need data"

    train_data, val_data = load_data(train_dir, val_dir)

    model = VGGModel()
    model.load_weights('vgg16_imagenet.h5')

    compile_model(model)

    history = train_model(model, train_data, val_data, epochs=5)
    
    model.save_weights('vgg_new_weights.h5')

    return model

model = run()
        
classes = [
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



# example image for detecting
img_path = "data/test/images/DSC_5941_JPG_jpg.rf.7f34ef03affd2f952f6519e8506d8cdc.jpg"

run_detection_and_visualization(img_path, model, classes)
