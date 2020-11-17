import numpy as np
import matplotlib.pyplot as plto_categoricalt

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop, SGD
from dataset_load import load_data, get_label

(X_train, Y_train), (X_test, Y_test)= load_data(test_size = 0.2, path='/content/dataset/')

drop_prob = 0.6

ks=(7,7)# kernel size
modelo = Sequential()
modelo.add(Conv2D(input_shape=[64,64,3],filters=32,kernel_size=ks,padding='same',activation='relu'))
modelo.add(MaxPooling2D())
modelo.add(Conv2D(input_shape=[32,32,3],filters=64,kernel_size=ks, padding='same',activation='relu'))
modelo.add(MaxPooling2D())
modelo.add(Flatten())
modelo.add(Dense(units=1024, activation='relu'))
modelo.add(Dropout(rate=drop_prob))
modelo.add(Dense(units=3, activation='softmax'))
modelo.summary()

epocas = 25
bs = 200
lr = 0.001
optim = Adam(lr)
modelo.compile(loss = 'categorical_crossentropy',optimizer=optim,metrics=['accuracy'])

historico = modelo.fit(X_train, Y_train, epochs=epocas,batch_size=bs, validation_split=0.2)

metricas = modelo.evaluate(X_test,Y_test,verbose=0)
print("Accuracy {:5.3}".format(metricas[1]))


