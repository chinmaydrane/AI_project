import os
import csv
import numpy as np
import tensorflow as tf
import random
from preprocessing import preprocess_image_inference, RESIZE_SHAPE

IMG_HEIGHT, IMG_WIDTH = RESIZE_SHAPE
NUM_CLASSES = 2
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 1
BATCH_SIZE_TEST = 1
AUGMENTATION_CHANCE = 1.0

class TBNetDSI:
    def __init__(self, data_path='data/'):
        self.data_path = data_path

    def parse_function(self, filename, label):
        """Preprocessing for val/test"""
        def _process(path):
            path = path.numpy().decode('utf-8')  # decode tensor to string
            image = preprocess_image_inference(path)
            return image.astype(np.float32)
        
        image = tf.py_function(func=_process, inp=[filename], Tout=tf.float32)
        image.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
        return {'image': image, 'label/one_hot': tf.one_hot(label, NUM_CLASSES)}

    def parse_function_train(self, filename, label):
        """Preprocessing + optional augmentation for training"""
        def _process(path):
            path = path.numpy().decode('utf-8')
            image = preprocess_image_inference(path)
            return image.astype(np.float32)

        image = tf.py_function(func=_process, inp=[filename], Tout=tf.float32)
        image.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))

        # Data augmentation
        def augment(img):
            if random.random() < AUGMENTATION_CHANCE:
                choice = random.randint(0, 3)
                if choice == 0:
                    img = tf.image.random_flip_left_right(img)
                elif choice == 1:
                    img = tf.image.random_brightness(img, 0.1)
                elif choice == 2:
                    img = tf.image.random_contrast(img, 0, 0.2)
            return img

        image = augment(image)
        return {'image': image, 'label/one_hot': tf.one_hot(label, NUM_CLASSES)}

    def get_split(self, csv_path, phase="train"):
        data_x, data_y = [], []
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)  # skip header
            for row in reader:
                data_x.append(os.path.join(self.data_path, row[0]))
                data_y.append(int(row[1]))

        dataset = tf.data.Dataset.from_tensor_slices((np.array(data_x), np.array(data_y)))

        if phase == "train":
            dataset = dataset.shuffle(5000).map(self.parse_function_train).repeat().batch(BATCH_SIZE_TRAIN)
            batch_size = BATCH_SIZE_TRAIN
        elif phase == "val":
            dataset = dataset.map(self.parse_function).batch(BATCH_SIZE_VAL)
            batch_size = BATCH_SIZE_VAL
        else:  # test
            dataset = dataset.map(self.parse_function).batch(BATCH_SIZE_TEST)
            batch_size = BATCH_SIZE_TEST

        return dataset, len(data_y), batch_size

    def get_train_dataset(self):
        return self.get_split('train.csv', 'train')

    def get_validation_dataset(self):
        return self.get_split('val.csv', 'val')

    def get_test_dataset(self):
        return self.get_split('test.csv', 'test')
