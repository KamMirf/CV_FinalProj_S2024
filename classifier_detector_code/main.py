import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from classifier_to_detector import VGGModel
from preprocess import Datasets

from tensorboard_utils import \
        ImageLabelingLogger, CustomModelSaver
from matplotlib import pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" 
python3 main.py --load-checkpoint checkpoints/vgg_model/050324-233826/vgg.e001-acc0.7443.weights.h5 --detect data/train/images/DSC_5691_JPG_jpg.rf.2d8253758a92224d7b732a70d06985e4.jpg
python3 main.py --load-checkpoint checkpoints/vgg_model/050324-233826/vgg.e001-acc0.7443.weights.h5 --detect data/test/images/DSC_5941_JPG_jpg.rf.7f34ef03affd2f952f6519e8506d8cdc.jpg
^^this weight outputs extremely high probabilities for spinach

Let's try a lower accuracy weight file

"""


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data',
        default='cropped_data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file.''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--detect',
        default=None,
        help='''Path to the image with which to run the classifier 
        as a detector on by sliding it over the image'''
    )

    return parser.parse_args()

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

# different scaled images
def image_pyramid(image, scale=0.5, min_size=(224, 224)): 
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
def object_detection(image, model, scale=1.5, win_size=(224, 224), step_size=32, threshold=0.99):
    boxes = []
    confidences = []
    classIDs = []

    # Create an image pyramid and sliding a window
    for resized in image_pyramid(image, scale, min_size=win_size):
        for (x, y, window) in sliding_window(resized, step_size=step_size, window_size=win_size):
            # if window.shape[0] != win_size[1] or window.shape[1] != win_size[0]:
            #     continue
            # Preprocess the window for classification
            window = tf.expand_dims(window, axis=0)
            window = tf.image.resize(window, win_size)
            preds = model.predict(window)
            print(preds)
            # classes = tf.argsort(preds, direction='DESCENDING')
            # classID = classes[0][1] if classes[0][0] == 25 else classes[0][0]
           
            # classID = tf.cast(classID, tf.int64)
            
            
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



def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """
    
  
    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, 3, 5)
    ]


    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=16,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of main.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)
    if os.path.exists(ARGS.load_vgg):
        ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)

    # Run script from location of main.py
    os.chdir(sys.path[0])

    datasets = Datasets(ARGS.data, 3)
    
    
    model = VGGModel()
    checkpoint_path = 'classifier_detector_code'+os.sep+"checkpoints" + os.sep + \
        "vgg_model" + os.sep + timestamp + os.sep
    logs_path = 'classifier_detector_code'+os.sep+"logs" + os.sep + "vgg_model" + \
        os.sep + timestamp + os.sep
    model(tf.keras.Input(shape=(224, 224, 3)))

    # Print summaries for both parts of the model
    model.vgg16.summary()
    model.head.summary()

    # Load base of VGG model
    model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        
        model.head.load_weights(ARGS.load_checkpoint)

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    
    if ARGS.detect is not None:
        "running detection"
        run_detection_and_visualization(ARGS.detect, model=model, classes=classes)
        return
    
    if ARGS.evaluate:
        test(model, datasets.test_data)
        
    else:
        print("training")
        train(model, datasets, checkpoint_path, logs_path, init_epoch)


# Make arguments global
ARGS = parse_args()

main()
