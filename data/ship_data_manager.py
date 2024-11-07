import os

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_and_preprocess_image(img_path, image_size):
    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    return img_array
def load_and_preprocess_image_expand_dims(img_path, image_size):
    return np.expand_dims(load_and_preprocess_image(img_path, image_size), axis=0)


class ShipDataManager:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path

    def load_images_normalized_from_folder_with_label(self, image_size=(80, 80)):
        images = []
        labels = []
        for filename in os.listdir(self.data_folder_path):
            images.append(load_and_preprocess_image(os.path.join(self.data_folder_path, filename), image_size))
            labels.append(0 if filename.startswith('0') else 1)
        return np.array(images), np.array(labels)