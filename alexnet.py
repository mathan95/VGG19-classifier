# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 10:57:16 2018

@author: MathanP
"""

#KERAS
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
#from keras.utils import np_utils
#from keras.utils import to_categorical
#
#import numpy as np
#import matplotlib.pyplot as plt
#
##import theano
#from PIL import Image

from keras.preprocessing.image import ImageDataGenerator

# SKLEARN



#address
train_dir="C:/Users/Mathan/Documents/Senzmate/data1/train"
validation_dir="C:/Users/Mathan/Documents/Senzmate/data1/test1"


train_datagen = ImageDataGenerator(
      rescale=1./255,
#      rotation_range=20`,
#      width_shift_range=0.2,
#      height_shift_range=0.2,
#      horizontal_flip=True,
#      fill_mode='nearest'
        )
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# Change the batchsize according to your system RAM
train_batchsize = 100
val_batchsize = 10
 
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64,64),
        batch_size=train_batchsize,
        class_mode='categorical',
        shuffle=True)
 
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(64,64),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)


input_shape=[227,227,3]
nClasses=2
def createModel():
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3),activation = 'relu'))
    #model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
    
    return model

model1 = createModel()


model1.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])

model1.summary()

history = model1.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)
model1.save('all_freezed.h5')
#model1.evaluate(test_data, test_labels_one_hot)

    
        

    
