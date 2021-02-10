import numpy as np 
from keras.models import load_model
import matplotlib.pyplot as plt
from numpy import load
import pickle
import tensorflow as tf

#pickle_in = open('reconstruction.pkl', 'rb')
#reconstructor = pickle.load(pickle_in)

features_symbol = load('features_data.npy')
print(tf.constant(features_symbol))
print(features_symbol.shape)
features_symbol_predicted = load('features_predicted.npy')
print(tf.constant(features_symbol_predicted))
print(features_symbol_predicted.shape)

#print(features_symbol[10, :, :, 0])

#plt.imshow(features_symbol[5, :, :, 0], cmap='gray')
#plt.show()
print(np.min(features_symbol_predicted[17, :, :, 0]))
print(np.max(features_symbol_predicted[17, :, :, 0]))
plt.imshow(features_symbol_predicted[17, :, :, 0], cmap='gray')
plt.show()

#reconstruction = load_model('reconstruction_model')
