import cv2
import os
from mask import create_mask


folder_path = "dataset/male"

images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for i in range(len(images)):
    print("the path of the image is", images[i])
    create_mask(images[i])
