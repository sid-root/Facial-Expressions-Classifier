import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

trdirpath="C:/Emojify/train"
tedirpath="C:/Emojify/test"

trdg=ImageDataGenerator(rescale=1./255)
tedg=ImageDataGenerator(rescale=1./255)

trdata=trdg.flow_from_directory(trdirpath,target_size=(48,48),batch_size=32,color_mode="grayscale",class_mode="categorical")
tedata=tedg.flow_from_directory(tedirpath,target_size=(48,48),batch_size=32,color_mode="grayscale",class_mode="categorical")

model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(48,48,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.0001,decay=1e-5),metrics=["accuracy"])

model_info=model.fit_generator(trdata,steps_per_epoch=28709//32,epochs=30,validation_data=tedata,validation_steps=7178//32)

model.save('model.h5')

