# Face Mask Multi-Class Dataset

## dataset zip files

*dataset* is aumented dataset using our maskpy  

*dataset_original_faces* have faces extracted from [kaggle dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) 


```.py
from dataset_load_v3 import load_data, get_label, load_images

(X_train, y_train), (X_test,y_test) = load_data(test_size = 0.3, path = './dataset/', random_state)

x,y = load_raw(path = './dataset/')

#Return np.array
images = load_images(path='./imgs/')

#predict: 0 | 1 | 2 from np.argmax of prediction
#return string 'with_mask' | 'without_mask' | 'wrog_mask'
print(get_label(predict))
```
