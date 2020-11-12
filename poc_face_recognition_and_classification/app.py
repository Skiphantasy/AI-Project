"""
Face recognition and binary classification based on gender (woman/man)

Just to test what is needed to build a project with these specifications

Using Validation sub-dataset from https://www.kaggle.com/cashutosh/gender-classification-dataset as whole dataset due to huge size, computing times etc.
"""

from PIL import Image, ImageDraw
from get_model import get_model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
import cv2
import face_recognition
import numpy as np

# Code below throws an error so it's better to load the weights after recreating the neural network (get_model.py)
# model = load_model('trainedmodel.h5')

model = get_model()

model.load_weights('trainedmodel.h5')

# set up the input file
IMG_FILE = 'sample_3.jpg'

# open the picture 3 times, face recognition, convert to array so the neural net can do its thing, and draw final image using PIL
image = face_recognition.load_image_file(IMG_FILE)
image_opencv = cv2.imread(IMG_FILE)
pil_image = Image.open(IMG_FILE)
draw = ImageDraw.Draw(pil_image)

f_location_1 = face_recognition.face_locations(image)
# loop over faces detected
for f in f_location_1:
	# detection box
	top, right, bottom, left = f
	detected = image_opencv[top:bottom, left:right]
	# pre-process pic
	detected = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
	detected = cv2.resize(detected, (224, 224))
	detected = img_to_array(detected)
	detected = preprocess_input(detected)
	detected = np.expand_dims(detected, axis=0)
	# actual classification
	(man, woman) = model.predict(detected)[0]
	label = "man" if man > woman else "woman"
	print("Recognized "+label)
	draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
	text_width, text_height = draw.textsize(label)
	draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
	draw.text((left + 6, bottom - text_height - 5), label, fill=(255, 255, 255, 255))
del draw

# display the resulting image
pil_image.show()
