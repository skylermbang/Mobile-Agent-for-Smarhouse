import csv
import os
import re  # get number in string
import math
import numpy as np
import pandas as pd
from keras.models import load_model, Model
from keras.layers import Dense, Input

# Following lines should be used when you training an new model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


# Read WAP, GEOMAG, LOC values
wap = pd.read_csv('wap_huawei.csv', header=None)
wap_list = wap.values.tolist()
wap_array = np.array(wap_list)

mag = pd.read_csv('geo_huawei.csv', header=None)
mag_list = mag.values.tolist()
mag_array = np.array(mag_list)

loc = pd.read_csv('loc_huawei.csv', header=None)
loc_list = loc.values.tolist()
loc_array = np.array(loc_list)

print('loc shape:', loc_array.shape)
print('geo shape:', mag_array.shape)
print('wap shape:', wap_array.shape)

# Normalize the WAP value
wap_array = wap_array / -110

# This is a line to load the trained model
# autoencoder = load_model('autoencoder.h5')

# This is the structure of the Auto-Encoder model, when you load the model, please invalidate the following lines
input = Input(shape=(516,))
encoding_dim = 4
encoded = Dense(128, activation='sigmoid')(input)
encoded = Dense(64, activation='sigmoid')(encoded)
encoded = Dense(10, activation='sigmoid')(encoded)
encoder_output = Dense(encoding_dim, activation='tanh')(encoded)

# This is the decoder part
decoded = Dense(10, activation='sigmoid')(encoder_output)
decoded = Dense(64, activation='sigmoid')(decoded)
decoded = Dense(128, activation='sigmoid')(decoded)
decoded = Dense(516, activation='sigmoid')(decoded)

# Compile the Auto-Encoder
autoencoder = Model(input=input, outputs=decoded) # This line should be invalidate when you loading a model
autoencoder.compile(optimizer='adam', loss='mse')

# Training the model
for step in range(100000):
    cost = autoencoder.train_on_batch(wap_array, wap_array)
    if step % 50 == 0:
        print('\nepoch:', step)
        print('cost:', cost)

# Save the Auto-Encoder model, be careful with the name of the model
autoencoder.save('autoencoder_2.h5')
