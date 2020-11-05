import csv
import os
import re  # get number in string
import math
import numpy as np
import pandas as pd
from keras.models import load_model, Model
import matplotlib.pyplot as plt
# Following lines should be used when you training an new model
from keras.models import Sequential
from keras.layers import *
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


# This is to load autoencoder, and this autoencoder will generate an 4 dimensions output
# But this program is begin with the full size data, so these lines are invalidated temporally
# autoencoder = load_model('autoencoder.h5')
# automodel = Model(inputs=autoencoder.input, outputs=autoencoder.layers[4].output)

# This is a line to load the CNN model
# model = load_model('CNN_model_2.h5')


# Normalize the data, all the value should in the bound [0,1]
Loc_x = loc_array[:, 0] / 50
Loc_y = loc_array[:, 1] / 24
# Combine the normalized data
Loc = np.vstack((Loc_x, Loc_y))

# Normalize the RSS value
RSS = wap_array / -110
# Get the encoded data from the Auto-Encoder, this line will be temporally invalidated
# RSS = automodel.predict(RSS)

# Expand the dimension to reach the requirements of the CNN
RSS = np.expand_dims(RSS, 2)
print('RSS_CNN shape:', RSS.shape)

# CNN model
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=4, activation='relu', strides=1, input_shape=(516, 1)))
model.add(Conv1D(100, 2, activation='relu'))
# model.add(MaxPooling1D(strides=2))
model.add(GlobalAveragePooling1D())
model.add(Dense(2, activation='relu'))
model.compile(loss='mse', optimizer='adam')


# Training procedure
for step in range(500):
    cost = model.train_on_batch(RSS, Loc.T)
    if step % 50 == 0:
        print('\nepoch:', step)
        print('cost:', cost)

# Generate the predict value and plot an image to see the result of training
pred_Loc_x = model.predict(RSS)
pred_loc_x, pred_loc_y = np.vsplit(pred_Loc_x.T, 2)

plt.scatter(Loc_x, Loc_y, edgecolor='g')
plt.scatter(pred_loc_x, pred_loc_y, edgecolors='r')
plt.show()
model.save('CNN_model_3.h5')
