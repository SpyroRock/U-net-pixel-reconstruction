from unet_model_construction import get_unet
import keras
from keras.models import Model
from keras.layers import Input
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow import multi_gpu_model
import keras
#from keras.models import Sequential, Model, model_from_json
#from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
#from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
from numpy import load, save
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
import pickle

img_height = 128
img_width = 128

img_height_test = 128
img_width_test = 128

speckle_data = load('speckle_cluster_case2.npy')
print(speckle_data.shape)
#img = cv2.imread('resized.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = img.reshape((img.shape[0],img.shape[1],1))
#speckle_labels = load('speckle_labels.npy')
speckle_labels = load('symbol_cluster_case2.npy')
print(speckle_labels.shape)
#plt.imshow(speckle_labels[2], cmap='gray')
#plt.show()
#dictionary = {speckle_labels_n: speckle_labels_mn_n for speckle_labels_n, speckle_labels_mn_n in zip(speckle_labels, speckle_labels_mn)}

X_train, X_test, y_train, y_test = train_test_split(speckle_data, speckle_labels, test_size=0.1, random_state=42)

X_train = X_train.reshape(-1, img_height, img_width, 1)
X_test = X_test.reshape(-1, img_height, img_width, 1)
input_shape = (img_height, img_width, 1)

y_train = y_train.reshape(-1, img_height_test, img_width_test, 1)
y_test = y_test.reshape(-1, img_height_test, img_width_test, 1)
input_shape_test = (img_height_test, img_width_test, 1)

print(X_train.shape)
print(y_train.shape)

# input_layer = Input((img_height, img_width, 1))
# reconstruction = build_model(input_layer, 16)

# reconstruction = create_model(pretrained_weights = None, input_size = input_shape, start_neurons = 64)

# reconstruction = unet_model(n_classes=1, 
#                             im_size=32, 
#                             n_channels=1, 
#                             n_filters_start=32, 
#                             growth_factor=2, 
#                             upconv=True)
# print(reconstruction)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

reconstruction = get_unet(input_shape, n_filters = 32, dropout = 0.1, batchnorm = True)

reconstruction.fit(X_train, y_train, 
                   batch_size = 10, 
                   epochs = 10500, 
                   verbose = 1, 
                   validation_data = (X_test, y_test)) # Data on which to evaluate the loss and any model metrics at the end of each epoch. 
                                                       # The model will not be trained on this data. 
                                                       # This could be a list (x_val, y_val) or a list (x_val, y_val, val_sample_weights). 
                                                       # validation_data will override validation_split.

score = reconstruction.evaluate(X_test, y_test, verbose = 0)

print('Test loss:', score[0])
print('Test acuracy:', score[1]) 

y_predicted = reconstruction.predict(X_test)

extract = Model(reconstruction.inputs, reconstruction.layers[-1].output) 
features = extract.predict(X_test)
print(features.shape)
save('features_data.npy', features)
save('features_predicted.npy', y_predicted)

reconstruction.save('reconstruction_model')

