import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
import hyperparameters as hp
from classifier_to_detector import VGGModel, CustomModel
from extract_results import process_image, run_detection_and_visualization
from preprocess import Datasets

from tensorboard_utils import \
        ImageLabelingLogger, CustomModelSaver
from matplotlib import pyplot as plt



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" 
VGG:
python3 main.py --model VGG --load-checkpoint checkpoints/vgg_model/050324-233826/vgg.e001-acc0.7443.weights.h5 --detect data/test/images/DSC_5941_JPG_jpg.rf.7f34ef03affd2f952f6519e8506d8cdc.jpg

Custom:
python3 main.py --model Custom --load-checkpoint checkpoints/Custom/050724-130938/custom.e010-acc0.8951.weights.h5 --detect data/test/images/DSC_5941_JPG_jpg.rf.7f34ef03affd2f952f6519e8506d8cdc.jpg
#does the best



"""


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        default=None,
        help='Choose classifer: VGG or Custom'
    ),
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

# # different scaled images
# def image_pyramid(image, scale=1.5, min_size=(224, 224)): 
#     yield image
#     while True:
#         w = int(image.shape[1] / scale)
#         image = tf.image.resize(image, (w, w))
#         if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
#             break
#         yield image

# # sliding window across image
# def sliding_window(image, step_size, window_size):

#     for y in range(0, image.shape[0] - window_size[1], step_size):
#         for x in range(0, image.shape[1] - window_size[0], step_size):
#             window = (image[y:y + window_size[1], x:x + window_size[0]])
#             yield (x, y, window)


# def object_detection(image, model, scale=1.5, win_size=(224, 224), step_size=32, threshold=0.5, max_box_area=0.25):
#     boxes = []
#     confidences = []
#     classIDs = []

#     # Maximum area of the box as a fraction of the image area
#     max_area_px = image.shape[0] * image.shape[1] * max_box_area

#     # Create an image pyramid and sliding a window
#     for resized in image_pyramid(image, scale, min_size=win_size):
#         for (x, y, window) in sliding_window(resized, step_size=step_size, window_size=win_size):

#             if x <= 30 or x >= image.shape[0] - 30 or y <= 30 or y >= image.shape[1] - 30:
#                 continue
#             window = tf.expand_dims(window, axis=0)
#             window = tf.image.resize(window, win_size)
#             preds = model.predict(window)
#             classID = tf.argmax(preds[0])
#             confidence = preds[0][classID]

#             if confidence > threshold:
#                 scale_x = image.shape[1] / resized.shape[1]
#                 scale_y = image.shape[0] / resized.shape[0]
#                 box = [x * scale_x, y * scale_y, (x + win_size[0]) * scale_x, (y + win_size[1]) * scale_y]
#                 box_width = box[2] - box[0]
#                 box_height = box[3] - box[1]
#                 box_area = box_width * box_height

#                 if box_area <= max_area_px:
#                     boxes.append(box)
#                     confidences.append(float(confidence))
#                     classIDs.append(classID.numpy())

#     if boxes:
#         boxes = tf.constant(boxes, dtype=tf.float32)  # Ensure boxes are a 2D Tensor
#         confidences = tf.constant(confidences, dtype=tf.float32)

#         idxs = tf.image.non_max_suppression(boxes, confidences, max_output_size=50, iou_threshold=0.5)
#         final_boxes = [boxes[i].numpy().tolist() for i in idxs]  # Convert tensor indices to list of boxes
#     else:
#         final_boxes = []

#     return final_boxes, confidences.numpy().tolist(), classIDs


# def draw_boxes(image, boxes, confidences, classIDs, classes):
#     for (box, score, classID) in zip(boxes, confidences, classIDs):
#         (startX, startY, endX, endY) = box
#         startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
#         label = f"{classes[classID]}: {score:.2f}"
#         cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
#         y = startY - 15 if startY - 15 > 15 else startY + 15
#         cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return image


# def run_detection_and_visualization(image_path, model, classes):
#     # Load and preprocess the image
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     wind_sz = (224, 224) if ARGS.model == "VGG" else (hp.window_size, hp.window_size)

#     # Perform object detection
#     final_boxes, confidences, classIDs = object_detection(image, model, scale=1.5, win_size=wind_sz, step_size=64, threshold=0.99, max_box_area=0.08)

#     # Draw bounding boxes on the image
#     output_image = draw_boxes(image.copy(), final_boxes, confidences, classIDs, classes)

#     # Display the image
#     # plt.figure(figsize=(10, 8))
#     # plt.imshow(output_image)
#     # plt.axis('off')
#     # plt.show()
#     return final_boxes, confidences, classIDs



def train(model, datasets, checkpoint_path, logs_path, init_epoch, model_type):
    """ Training routine. """
    
  
    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, model_type=model_type, max_num_weights=5)
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

    
    
    
    
    if ARGS.model is None:
        print("Error: Must select a model (VGG or Custom)")
        return
    elif ARGS.model == "VGG":
        model = VGGModel()
        checkpoint_path = 'classifier_detector_code'+os.sep+"checkpoints" + os.sep + \
        "vgg_model" + os.sep + timestamp + os.sep
        logs_path = 'classifier_detector_code'+os.sep+"logs" + os.sep + "vgg_model" + \
        os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))
        
    elif ARGS.model == "Custom":
        model = CustomModel()
        checkpoint_path = "checkpoints" + os.sep + \
        "Custom" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "Custom" + \
        os.sep + timestamp + os.sep #'classifier_detector_code'+os.sep+
        model(tf.keras.Input(shape=(hp.window_size, hp.window_size, 3)))
        
    else:
        print("Error: VGG or Custom only")
        return 


    if ARGS.model == "VGG" and ARGS.load_checkpoint:
        # Load base of VGG model
        model.vgg16.summary()
        model.head.load_weights(ARGS.load_checkpoint)
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)
    elif (ARGS.model == "Custom" and ARGS.load_checkpoint):
        model.load_weights(ARGS.load_checkpoint)
    elif (ARGS.model == "Custom"):
        model.body.summary()
    model.head.summary()



    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    
    if ARGS.detect is not None:
       
        final_boxes, confidences, classIDs = run_detection_and_visualization(ARGS.detect, model=model, classes=classes, model_type=ARGS.model)
        results = process_image(final_boxes, confidences, classIDs)
        return results
        # example return can be seen below
    datasets = Datasets(ARGS.data, ARGS.model)
    if ARGS.evaluate:
        test(model, datasets.test_data)  
    else:
        print("training")
        train(model, datasets, checkpoint_path, logs_path, init_epoch, ARGS.model)


# Make arguments global
ARGS = parse_args()

main()

""" 
(Don't forget to include --load-checkpoint)
Example results after including --detect path/to/image

{
    "chocolate": {
        "count": 6,
        "boxes": [
            [512.0, 192.0, 542.0, 222.0],
            [256.0, 512.0, 286.0, 542.0],
            [256.0, 128.0, 286.0, 158.0],
            [
                576.9014282226562,
                192.30047607421875,
                621.9718017578125,
                237.3708953857422,
            ],
            [256.0, 256.0, 286.0, 286.0],
            [
                384.6009521484375,
                288.4507141113281,
                429.6713562011719,
                333.5211181640625,
            ],
        ],
        "confidences": [
            1.0,
            1.0,
            0.9999668598175049,
            0.9989758729934692,
            0.999701201915741,
            0.9999998807907104,
        ],
    },
    "flour": {
        "count": 1,
        "boxes": [
            [
                192.30047607421875,
                96.15023803710938,
                237.3708953857422,
                141.2206573486328,
            ]
        ],
        "confidences": [0.9953314661979675],
    },
    "ham": {
        "count": 6,
        "boxes": [
            [320.0, 128.0, 350.0, 158.0],
            [384.0, 128.0, 414.0, 158.0],
            [448.0, 128.0, 478.0, 158.0],
            [
                288.4507141113281,
                144.22535705566406,
                356.05633544921875,
                211.8309783935547,
            ],
            [
                288.4507141113281,
                288.4507141113281,
                333.5211181640625,
                333.5211181640625,
            ],
            [
                216.71957397460938,
                216.71957397460938,
                318.306884765625,
                318.306884765625,
            ],
        ],
        "confidences": [
            0.9968271851539612,
            0.9999899864196777,
            1.0,
            1.0,
            1.0,
            0.9999992847442627,
        ],
    },
    "milk": {
        "count": 7,
        "boxes": [
            [
                480.75115966796875,
                384.6009521484375,
                525.8215942382812,
                429.6713562011719,
            ],
            [448.0, 384.0, 478.0, 414.0],
            [
                433.43914794921875,
                216.71957397460938,
                535.0264282226562,
                318.306884765625,
            ],
            [512.0, 256.0, 542.0, 286.0],
            [
                192.30047607421875,
                288.4507141113281,
                237.3708953857422,
                333.5211181640625,
            ],
            [128.0, 64.0, 158.0, 94.0],
            [448.0, 256.0, 478.0, 286.0],
        ],
        "confidences": [
            0.9999551773071289,
            0.9986794590950012,
            0.999990701675415,
            0.9924749732017517,
            1.0,
            1.0,
            1.0,
        ],
    },
    "spinach": {
        "count": 1,
        "boxes": [
            [
                432.6760559082031,
                144.22535705566406,
                500.28167724609375,
                211.8309783935547,
            ]
        ],
        "confidences": [0.9938913583755493],
    },
    "sugar": {
        "count": 2,
        "boxes": [
            [
                288.4507141113281,
                288.4507141113281,
                356.05633544921875,
                356.05633544921875,
            ],
            [
                216.71957397460938,
                433.43914794921875,
                318.306884765625,
                535.0264282226562,
            ],
        ],
        "confidences": [0.9957306981086731, 0.9995453953742981],
    },
}

"""