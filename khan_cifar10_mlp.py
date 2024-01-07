"""
    Importing necessary libraries
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
import keras
from keras.models import Sequential
from keras.layers import Dense

"""
    Defining training data reading function to unpickle
"""

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

"""
    Reading training data using defined function
"""

batch_1=unpickle("data_batch_1")
batch_2=unpickle("data_batch_2")
batch_3=unpickle("data_batch_3")
batch_4=unpickle("data_batch_4")
batch_5=unpickle("data_batch_5")

"""
    Combining data using list
"""

train_x = list(batch_1["data"])+list(batch_2["data"])+list(batch_3["data"])+list(batch_4["data"])+list(batch_5["data"])
train_y = list(batch_1["labels"])+list(batch_2["labels"])+list(batch_3["labels"])+list(batch_4["labels"])+list(batch_5["labels"])

"""
    making them numpy array for further operation
"""

train_x = np.array(train_x)
train_y = np.array(train_y)

"""
    using standard normal formula for normalization
    ddof 1 is used because we are treating them as sample
"""

standardized_train_x = (train_x - np.mean(train_x)) / (np.std(train_x, ddof=1))

"""
    Defining models
"""

model = Sequential()
model.add(Dense(64, input_dim = 3072, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

"""
    I have taken learning rate as 0.005
"""

model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.005), loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])

"""
    A total of 70% data will be trained
"""

a = model.fit(standardized_train_x, train_y, epochs = 20, batch_size = 128, validation_split = 0.3)

"""
    Playing with test data with the same manner
"""

test_data = unpickle("test_batch")
test_x = np.array(test_data["data"])
test_y = np.array(test_data["labels"])
standardized_test_x = (test_x - np.mean(test_x)) / (np.std(test_x, ddof=1))
_ , test_accuracy = model.evaluate(standardized_test_x, test_y)
print(f'Test accuracy is {round(100 * test_accuracy,4)} percent')

"""
    Learning graph
"""

plt.figure()
plt.plot(a.history['loss'])
plt.plot(a.history['val_loss'])
plt.title("Learning Model")
plt.ylabel("Loss Value from Model Fit")
plt.xlabel("Epoch")
plt.legend(['Train Loss', 'Validation Loss'])
plt.show()