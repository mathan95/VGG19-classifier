# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:24:10 2018

@author: MathanP
"""

from keras.applications import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
#from keras.applications.vgg19 import decode_predictions
import matplotlib.pyplot as plt
import numpy as np

# load the model
model = InceptionV3()
# load an image from file
image = load_img('vg2.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
plt.imshow(image)
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)


print(np.sum(yhat[0,300:323]))
# convert the probabilities to class labels
#label = decode_predictions(yhat)
## retrieve the most likely result, e.g. highest probability
#label1=label[0]
#n=len(label1)
#print (label1)
#for i in range(0,n,1):
#    label2=label1[i]
#    # print the classification
#    print('%s (%.2f%%)' % (label2[1], label2[2]*100))
