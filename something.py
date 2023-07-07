import tensorflow as tf
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

train_dir = "archive\\train"


def made_data_pipeline():
    data = tf.keras.utils.image_dataset_from_directory(train_dir)
    data_iterator = data.as_numpy_iterator()

made_data_pipeline()