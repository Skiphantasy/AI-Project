# Face Mask Multi-Class Dataset

## dataset zip files

*dataset* is aumented dataset using our maskpy  

*dataset_original_faces* have faces extracted from [kaggle dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) 


```.py
from dataset_load import load_data

(X_train, y_train), (X_test,y_test) = load_data(test_size = 0.3, path = './dataset/')

x,y = load_raw(path = './dataset/')

```
