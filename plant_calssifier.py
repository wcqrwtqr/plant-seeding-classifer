#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 22:52:36 2018

@author: mohammedalbatati
"""

import glob
#import os
from PIL import Image, ImageOps
import numpy as np
#from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import cv2
import pandas as pd
import timeit

'''
Specail thanks to the following fellow:
    ASHISHKUMAR ,https://www.kaggle.com/myashish/plant-seedling-classification
    Xingyu Yang , https://www.kaggle.com/xingyuyang/cnn-with-keras
    Gábor VecseiPlant,https://www.kaggle.com/gaborvecsei/plant-seedlings-fun-with-computer-vision
'''

start_timing = timeit.timeit()

## Helper function to sharpen the images
# special thanks to ASHISHKUMAR ,https://www.kaggle.com/myashish/plant-seedling-classification
def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


# The below code uses the convert functions to sharpen the images
# convert the images to sharper images 
#z = glob.glob('train/*/*.png')
#ori_label = []
#ori_imgs = []
#for fn in z:
#    if fn[-3:] != 'png':
#        continue
#    ori_label.append(fn.split('/')[-2])
#    img = cv2.imread(fn)
#    seg_img = segment_plant(img)
#    imge_write = cv2.imwrite(fn,seg_img)
#    new_img = Image.open(fn)
#    ori_imgs.append(ImageOps.fit(new_img, (48, 48), Image.ANTIALIAS).convert('RGB'))

# The below code uses the normal image as is
#z = glob.glob('train back up/*/*.png')
#ori_label = []
#ori_imgs = []
#for fn in z:
#    if fn[-3:] != 'png':
#        continue
#    ori_label.append(fn.split('/')[-2])
#    new_img = Image.open(fn)
#    ori_imgs.append(ImageOps.fit(new_img, (48, 48), Image.ANTIALIAS).convert('RGB'))


# The below code uses the normal image as is
# load the images that been sharpened
z = glob.glob('train/*/*.png')
ori_label = []
ori_imgs = []
print('Start reading the images to memory')
for fn in z:
    if fn[-3:] != 'png':
        continue
    ori_label.append(fn.split('/')[-2])
    new_img = Image.open(fn)
    ori_imgs.append(ImageOps.fit(new_img, (48, 48), Image.ANTIALIAS).convert('RGB'))

load_images_timing = timeit.timeit()
print('Loading images to memory: ',load_images_timing - start_timing)


print('label binarizing')
# convert the images to numpy array and store it in a list for later splitting
    #converting the array to shape 48 x 48 x 3 channel for the keras learning
    # converting the catergories to label of [0,0,0,0,1,0,0,0,0]
imgs = np.array([np.array(im) for im in ori_imgs])
imgs = imgs.reshape(imgs.shape[0], 48, 48, 3) / 255
lb = LabelBinarizer().fit(ori_label)
label = lb.transform(ori_label)


# splitting the data 
trainX, validX, trainY, validY = train_test_split(imgs, label, test_size=0.2, random_state=42)

# CNN modelling
from keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(48,3,3, input_shape = (48, 48, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16,3,3,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dense(12, activation='softmax'))
model.summary()

# compiling the model
model.compile(Adam(lr=0.0001) , loss='categorical_crossentropy',metrics=['accuracy'])


from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint

batch_size = 100
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
earlystop = EarlyStopping(patience=10)
modelsave = ModelCheckpoint(filepath='model_sharpened_images_extralayer.h5', save_best_only=True, verbose=1)
train_start_timing = timeit.timeit()
# training the model
model.fit(trainX, trainY,
          batch_size=batch_size,
          epochs=200,
          validation_data=(validX, validY),
          callbacks=[annealer, earlystop, modelsave])

train_end_timing = timeit.timeit()
print('Total Training timing: ',train_end_timing - train_start_timing)


## Sharpent the images and load them to memory
#print('sharpen the images')
#z = glob.glob('test_tobe_sharpend/*.png')
#test_imgs = []
#names = []
#for fn in z:
#    if fn[-3:] != 'png':
#        continue
#    names.append(fn.split('/')[-1])
#    img = cv2.imread(fn)
#    seg_img = segment_plant(img)
#    imge_write = cv2.imwrite(fn,seg_img)
#    new_img = Image.open(fn)
#    test_img = ImageOps.fit(new_img, (48, 48), Image.ANTIALIAS).convert('RGB')
#    test_imgs.append(test_img)
##from keras.models import Model, load_model
##model = load_model('model.h5')

load_test_images_timing_start = timeit.timeit()
# Load the images to memory
print('Loading the images')
z = glob.glob('test_tobe_sharpend/*.png')
test_imgs = []
names = []
for fn in z:
    if fn[-3:] != 'png':
        continue
    names.append(fn.split('/')[-1])
    new_img = Image.open(fn)
    test_img = ImageOps.fit(new_img, (48, 48), Image.ANTIALIAS).convert('RGB')
    test_imgs.append(test_img)
#from keras.models import Model, load_model
#model = load_model('model.h5')

load_test_images_timing = timeit.timeit()
print('Loading test images to memory: ',load_test_images_timing - load_test_images_timing_start)


timgs = np.array([np.array(im) for im in test_imgs])
testX = timgs.reshape(timgs.shape[0], 48, 48, 3) / 255

print('Predicting the labels and submitting the data')

yhat = model.predict(testX)
test_y = lb.inverse_transform(yhat)

df = pd.DataFrame(data={'file': names, 'species': test_y})
df_sort = df.sort_values(by=['file'])
df_sort.to_csv('results.csv', index=False)

print('final file submitted')

end_timing = timeit.timeit()
print('Total running time: ',end_timing - start_timing)





