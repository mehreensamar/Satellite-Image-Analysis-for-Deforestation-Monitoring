import cv2
import os
import numpy as np
from config import IMAGE_SIZE

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img

def load_dataset(folder_path):
    images = []
    labels = []
    for label in os.listdir(folder_path):
        class_path = os.path.join(folder_path, label)
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            images.append(preprocess_image(img_path))
            labels.append(label)
    return np.array(images), np.array(labels)
