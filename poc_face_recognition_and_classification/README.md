# PoC of Binary Classification

This is a test of binary classification using CNN and [Face-Recognition API](https://pypi.org/project/face-recognition/) in order to detect whether a human face is "man" or "woman".

## Usage:
```
conda create -n <venv-name> python=3.6 anaconda
source activate <venv-name>
pip install -r requirements.txt
python app.py
```

Current accuracy of the model is about 91%, but needs some work.

## Scripts

`loop_through_folder.py` and `mask.py` are helpful scripts located in [observations's repo by prajnasb](https://github.com/prajnasb/observations/blob/master/mask_classifier/Data_Generator) which also uses [Face-Recognition API](https://pypi.org/project/face-recognition/) to put a face mask from png file over the mouth of the face detection, saving the result to another file. This is used to expand the base dataset.