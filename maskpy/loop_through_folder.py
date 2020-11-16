import cv2
import os
from mask import create_mask


folder_path = "input"
save_path = "output"

images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

start = 0
end = 100

for i in range(start, end):
    print("the path of the image is", images[i])
    create_mask(images[i],save_path,'normal')


for i in range(start, end):
    os.remove(images[i])