import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import hyperparameters as hp


class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path, model_type):

        self.data_path = data_path
        self.model_type = model_type

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * 30

        # Mean and std for standardization
        self.mean = np.zeros((224, 224, 3))
        self.std = np.ones((224,224,3))
        self.calc_mean_and_std()

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), is_vgg=model_type=='VGG', shuffle=True, augment=True, testing=False)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), is_vgg=model_type=='VGG', shuffle=False, augment=False, testing=True)
 
        

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:400]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (400, 224, 224, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((224, 224))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img


        self.mean = np.mean(data_sample, axis=0)
        self.std = np.std(data_sample, axis=0)


        print("Dataset mean shape: [{0}, {1}, {2}]".format(
            self.mean.shape[0], self.mean.shape[1], self.mean.shape[2]))

        print("Dataset mean top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0,0,0], self.mean[0,0,1], self.mean[0,0,2]))

        print("Dataset std shape: [{0}, {1}, {2}]".format(
            self.std.shape[0], self.std.shape[1], self.std.shape[2]))

        print("Dataset std top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0,0,0], self.std[0,0,1], self.std[0,0,2]))

    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """

        img = (img - self.mean) / self.std  


        return img

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """

        img = tf.keras.applications.vgg16.preprocess_input(img)
      
        return img

    def custom_preprocess_fn(self, img):
        """ Custom preprocess function for ImageDataGenerator. """

        img = tf.keras.applications.vgg16.preprocess_input(img)

        return img

    def get_data(self, path, is_vgg, shuffle, augment, testing):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            is_vgg - Boolean value indicating whether VGG preprocessing
                     should be applied to the images.
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        if augment:

            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn
                , 
                channel_shift_range=20,
                brightness_range=[0.8,1.2],
                rotation_range=20, 
                width_shift_range=0.15, 
                height_shift_range=0.15,
                zoom_range=0.10,
                horizontal_flip=True
                )

            # ============================================================
        else:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        # VGG must take images of size 224x224
        img_size = 224 if is_vgg else hp.window_size

        classes_for_flow = None
        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=100,
            shuffle=shuffle,
            classes=classes_for_flow)

        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class
                
        return data_gen
        # Parse image paths and corresponding labels
        # images = []
        # labels = []
        # bounding_boxes = []  # New list to store bounding box information
        # for root, _, files in os.walk(path):
        #     for name in files:
        #         if name.endswith(".jpg"):
        #             img_path = os.path.join(root, name)
        #             images.append(img_path)
                    
        #             if (not testing):
        #                 label_path = img_path.replace("train/images", "train/labels").replace(".jpg", ".txt")
        #             elif (testing):
        #                 label_path = img_path.replace("test/images", "test/labels").replace(".jpg", ".txt")
                    
                    
        #             with open(label_path, "r") as f:
        #                 label = f.read().strip().split()
        #                 labels.append(int(label[0]))  # Assuming class is the first number
        #                 # Read bounding box information from label file
        #                 box_info = [float(val) for val in label[1:]]  # Assuming bounding box info follows the class
        #                 bounding_boxes.append(box_info)

        # # Shuffle data
        # # if shuffle:
        # #     combined = list(zip(images, labels))
        # #     random.shuffle(combined)
        # #     images, labels = zip(*combined)

        # # Setup the dictionaries if not already done
        # self.classes = [i for i in range(30)]
        # if not bool(self.idx_to_class):
        #     # self.classes = list(set(labels))
        #     self.classes.sort()
        #     for i, img_class in enumerate(self.classes):
        #         self.idx_to_class[i] = img_class
        #         self.class_to_idx[img_class] = i
        # print(f'index to class dict: {self.idx_to_class}\n')
        # print(f'class to index dict: {self.class_to_idx}\n')
        # print(f'classes: {self.classes}\n')
        # # Convert class labels to indices
        # label_indices = [self.class_to_idx[label] for label in labels]
        # print(f'label_indices: {label_indices}\n')
        # # bounding_boxes = np.array(bounding_boxes)
        # print(f'bounding boxes {bounding_boxes}\n')

        # data_gen = data_gen.flow_from_dataframe(
        #     dataframe=pd.DataFrame({"filepath": images, "class": label_indices}),
        #     x_col="filepath",
        #     y_col="class",
        #     target_size=(img_size, img_size),
        #     class_mode='raw',
        #     batch_size=10,
        #     shuffle=shuffle
        # )
        
        # return data_gen, bounding_boxes
