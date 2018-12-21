import os

from tensorflow.python.keras.applications import ResNet50
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array
from PIL import Image
from IPython.display import display
from os.path import join

from decode_predictions import decode_predictions

image_dir = '/Users/anupamchowdhury/PycharmProjects/ExtendedResNet50DogBreedIdentificationKaggle/data/'
img_paths = [join(image_dir , filename) for filename in [
    'fb5898e240410c7d736548bf938bbc0a.jpg',
    'fdccec2dc716306a12b773e7689887c0.jpg',
    'ffa0055ec324829882186bae29491645.jpg',
    'ffcde16e7da0872c357fbc7e2168c05f.jpg']]

print(img_paths)

image_size = 224

def read_and_prep_images(img_paths, img_height = image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height,img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return output

model_path = '/Users/anupamchowdhury/PycharmProjects/ExtendedResNet50DogBreedIdentificationKaggle/data/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
if os.path.exists(model_path):
    print('model weights loading...')
my_model = ResNet50(weights = model_path)
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)

most_likely_labels = decode_predictions(preds, top=3, class_list_path='/Users/anupamchowdhury/PycharmProjects/ExtendedResNet50DogBreedIdentificationKaggle/data/imagenet_class_index.json')

for i, img_path in enumerate(img_paths):
    Image.open(img_path)
    print(most_likely_labels[i])