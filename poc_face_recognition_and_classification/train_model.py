from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


DATASET_DIR = 'dataset'
imgs = list(paths.list_images(DATASET_DIR))


X = []
y = []

print("Data pre-processing: ")
for img in imgs:
    label = img.split(os.path.sep)[1]
    
    image = load_img(img, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    
    X.append(image)
    y.append(label)


X = np.array(X, dtype="float32")
y = np.array(y)

lb = LabelBinarizer()
y = lb.fit_transform(y)
y = to_categorical(y)

print("Data splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)


aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)


for layer in baseModel.layers:
	layer.trainable = False

print("Establishing hyperparameters")
N_EPOCHS = 20
LR = 0.001
BS = 32

opt = Adam(lr=LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("Training...")
hist = model.fit(X_train, y_train, batch_size=BS, epochs=N_EPOCHS)


print("Testing model")
evaluated = model.evaluate(X_test, y_test)
print(evaluated)

print("Saving model...")
model.save('trainedmodel.h5', save_format="h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(hist.history['loss'], c='r', label='Loss Training')
plt.plot(hist.history['accuracy'], c='b', label='Accuracy Training')
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.savefig("graphs.png")

