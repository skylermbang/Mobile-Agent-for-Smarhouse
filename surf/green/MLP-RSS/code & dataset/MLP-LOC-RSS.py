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

# Load the model of Auto-Encoder (Trained in another code)
autoencoder = load_model('autoencoder.h5')

# This line could separate the encoder and decoder, 'automodel' means it only has encoder
automodel = Model(inputs=autoencoder.input, outputs=autoencoder.layers[4].output)

# Load the already trained model, when you want to train another model, please invalidate this line
model = load_model('MLP_model_1.h5')


# Normalize the data, all the value should in the bound [0,1]
Loc_x = loc_array[:, 0] / 50
Loc_y = loc_array[:, 1] / 24
# Combine the normalized data
Loc = np.vstack((Loc_x, Loc_y))

# Normalize the RSS value
RSS = wap_array / -110
# Get the encoded data from the Auto-Encoder
RSS = automodel.predict(RSS)

# You need to invalidate these following line when you using the model

# model = Sequential()
# model.add(Dense(units=20, input_dim=4))
# model.add(Activation('sigmoid'))
# model.add(Dense(units=1000, input_dim=200))
# model.add(Activation('sigmoid'))
# model.add(Dense(units=1000))
# model.add(Activation('sigmoid'))
# model.add(Dense(units=2))
# model.add(Activation('sigmoid'))

# This is to compile the model
model.compile(optimizer='adam', loss='mse')


# This is to train the model
for step in range(100000):
    cost = model.train_on_batch(RSS, Loc)
    if step % 50 == 0:
        print('\nepoch:', step)
        print('cost:', cost)


# This could get the predict result of the model
pred_Loc_x = model.predict(RSS)

# Following lines are separating the X and Y value
pred_loc_x, pred_loc_y = np.vsplit(pred_Loc_x.T, 2)

# Plot the X and Y value, there are two different color, 'green' and 'red', 'green' for reference points
#                                                        'red' for the predicted points
plt.scatter(Loc_x, Loc_y, edgecolor='g')
plt.scatter(pred_loc_x, pred_loc_y, edgecolors='r')
plt.show()
model.save('MLP_model_2.h5') # When you save an new model, notice that the name should not be same as the previous model
