
import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random

def load_images(folder, raw = False):
    images = []
    for filename in os.listdir(folder):
        
        img = cv2.imread(os.path.join(folder,filename),1)
        if img is not None:
            if not raw:
                img = cv2.resize(img,(64,64))
                
            if img is not None:
                images.append(img)


    images = [img/255. for img in images]

    return np.array(images)



def load_raw(path = './dataset/'):
    images = np.array([])
    labels = np.array([])
    label = 0
    
    
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        
        imgs = load_images(os.path.join(path,folder),True)
        
        if label == 0:
            images = np.array(imgs)
        else:
            images = np.concatenate([images,imgs])
        
        labels = np.concatenate([labels, np.array([str(folder)]*imgs.shape[0])])
        label += 1
    
    
    data = list(zip(images,labels))
    
    #random.seed(1234)
    
    random.shuffle(data)
    random.shuffle(data)
    
    x,y = zip(*data)
    x = np.array(x)
    y = np.array(y)
    
    return (x,y)

def load_data(test_size= 0.3, path = './dataset/'):
   
        
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
        
        labels = np.concatenate([labels,np.array([label]*imgs.shape[0])])
        label += 1
    
    
    data = list(zip(images,labels))
    
    #random.seed(1234)
    
    random.shuffle(data)
    random.shuffle(data)
    
    x,y = zip(*data)
    x = np.array(x)
    y = pd.get_dummies(y)
    y = np.array(y)

    size = x.shape[0]
    split_size =  size - int(size*test_size)
    
    X_train = x[:split_size]
    X_test = x[split_size:]
    Y_train = y[:split_size]
    Y_test = y[split_size:]
    
    
    return (X_train, Y_train), (X_test,Y_test)

def get_dicc():
  dicc = {0:"with_mask", 1:"without_mask", 2:"wrong_mask"}
  
  return dicc
  
def get_label(value):
  dicc = get_dicc()
  return dicc[np.argmax(value)]
