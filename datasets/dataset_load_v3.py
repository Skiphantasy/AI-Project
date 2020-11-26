
import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.model_selection import train_test_split

def load_images(folder):
    dim = (96,96)
    dim2 = (80,80)
    images = []
    
    files = os.listdir(folder)
    files.sort()
    for filename in files:
        
        img = cv2.imread(os.path.join(folder,filename),1)
        if img is not None:
            img = cv2.resize(img,dim)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

 
    # images = [img/255. for img in images]

    return np.array(images)



def load_data(test_size= 0.3, path = './dataset/', random_state = False):
    if not random_state:
      random_state = int(time.time())
        
    random.seed(int(random_state))
    
    images = np.array([])
    labels = np.array([])
    label = 0
    
    folders = os.listdir(path)
    folders.sort()

    for folder in folders:
        
        imgs = load_images(os.path.join(path,folder))

        if label == 0:
            images = np.array(imgs)
        else:
            images = np.concatenate([images,imgs])
        
        print("{} tiene {}".format(label,imgs.shape[0]))
        labels = np.concatenate([labels,np.array([label]*imgs.shape[0])])
        label += 1
      
   



    x = np.array(images)
    y = pd.get_dummies(labels)
    y = np.array(y)

    X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size = test_size, random_state = random_state)

    print("Random State: {}".format(random_state))
    return (X_train, Y_train), (X_test,Y_test)

def get_dicc():
  dicc = {0:"with_mask", 1:"without_mask", 2:"wrong_mask"}
  
  return dicc
  
def get_label(value):
  dicc = get_dicc()
  return dicc[np.argmax(value)]


