# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:18:01 2022

@author: Anirudh
"""

import tensorflow as tf
import os
import random
import numpy as np
import math
 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = "C:/Users/Anirudh/OneDrive/Desktop/stage1_train/"
TEST_PATH = "C:/Users/Anirudh/OneDrive/Desktop/stage1_test/"


train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
    break        
    Y_train[n] = mask

# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    break

print('Done!')

image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()


#Model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
theta_g=tf.keras.layers.Conv2D(128,(1,1),strides=(2,2),padding='same')(c4)
phi_g=tf.keras.layers.Conv2D(128,(1,1),padding='same')(c5)
u6 = tf.keras.layers.add([phi_g, theta_g])
activation_xg=tf.keras.layers.Activation('relu')(u6)
psi=tf.keras.layers.Conv2D(1,(1,1),padding='same')(activation_xg)
sigmoid_xg=tf.keras.layers.Activation('sigmoid')(psi)
upsample_psi=tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(sigmoid_xg)
y=tf.keras.layers.multiply([upsample_psi,c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(y)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)



theta_g_u7=tf.keras.layers.Conv2D(64,(1,1),strides=(2,2),padding='same')(c3)
phi_g_u7=tf.keras.layers.Conv2D(64,(1,1),padding='same')(c6)
u7 = tf.keras.layers.add([phi_g_u7, theta_g_u7])
activation_xg_u7=tf.keras.layers.Activation('relu')(u7)
psi_u7=tf.keras.layers.Conv2D(1,(1,1),padding='same')(activation_xg_u7)
sigmoid_xg_u7=tf.keras.layers.Activation('sigmoid')(psi_u7)
#print(np.shape(sigmoid_xg_u7))
sigma=np.std(sigmoid_xg_u7)
thr=sigma*((math.log(64,2)**1/2))
#if()
           
upsample_psi_u7=tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(sigmoid_xg_u7)
print(np.shape(upsample_psi_u7),np.shape(c3))
y_u7=tf.keras.layers.multiply([upsample_psi_u7,c3]) 
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(y_u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)



theta_g_u8=tf.keras.layers.Conv2D(32,(1,1),strides=(2,2),padding='same')(c2)
phi_g_u8=tf.keras.layers.Conv2D(32,(1,1),padding='same')(c7)
u8 = tf.keras.layers.add([phi_g_u8, theta_g_u8])
activation_xg_u8=tf.keras.layers.Activation('relu')(u8)
psi_u8=tf.keras.layers.Conv2D(1,(1,1),padding='same')(activation_xg_u8)
sigmoid_xg_u8=tf.keras.layers.Activation('sigmoid')(psi_u8)
upsample_psi_u8=tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(sigmoid_xg_u8)
y_u8=tf.keras.layers.multiply([upsample_psi_u8,c2])  
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(y_u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)



theta_g_u9=tf.keras.layers.Conv2D(16,(1,1),strides=(2,2),padding='same')(c1)
phi_g_u9=tf.keras.layers.Conv2D(16,(1,1),padding='same')(c8)
u9 = tf.keras.layers.add([phi_g_u9, theta_g_u9])
activation_xg_u9=tf.keras.layers.Activation('relu')(u9)
psi_u9=tf.keras.layers.Conv2D(1,(1,1),padding='same')(activation_xg_u9)
sigmoid_xg_u9=tf.keras.layers.Activation('sigmoid')(psi_u9)
upsample_psi_u9=tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(sigmoid_xg_u9)
y_u9=tf.keras.layers.multiply([upsample_psi_u9,c1])   
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(y_u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

#tf.keras.models.save_model(results)


cred=model.predict(X_test)
for i,img in enumerate(cred):
    cv2.imwrite(f'{i}.png', img)











