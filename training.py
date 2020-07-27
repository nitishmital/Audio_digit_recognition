import os
import scipy.io.wavfile as wavf
import sounddevice as sd
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

parent_folder="/home/nitish/PycharmProjects/AudioMNIST/recordings/"  # Replace with the path depending on your own file system


input_shape = (100, 101, 1)
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model = tf.keras.models.load_model(parent_folder+"AudioDiginet.h5")

N_epochs=10
for epoch in range(N_epochs):
    for batch in range(1000):   # iterating over 1000 mini batches for 1 epoch
        x_train=[]
        y_train=[]
        batch_size=10
        for iter in range(batch_size):
            folder_iter=np.random.random_integers(1,40)
            which_folder="0"+str(folder_iter)
            which_folder=which_folder[len(which_folder) - 2:len(which_folder)]
            folder=parent_folder+which_folder+"_csv/"
            list_files=sorted(os.listdir(folder))
            file_index=np.random.random_integers(1,500)
            file_path = folder + list_files[file_index-1]

            with open(folder+list_files[file_index-1], newline='') as csvfile:
                csv.field_size_limit(sys.maxsize)
                csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                row=next(csvreader)
                reshaped_row = np.reshape(row[1:], (501, 101))
                x_train.append(reshaped_row[:100].astype('float32')) # only take the first 100 frequency coefficients to reduce the size of the variables. Essentially low pass filtering for size reduction.
                y_train.append(int(float(row[0])))

        x_train = np.array(x_train)
        x_train /= 200  # Normalizing data to a smaller size

        x_train = x_train.reshape(x_train.shape[0], 100, 101, 1)
        model.fit(x=x_train,y=y_train, epochs=1)

# Save the model
model.save(parent_folder+"AudioDigiNet.h5")


'''
    f=list(range(100))
    t=list(range(101))
    plt.figure(figsize=(10, 6))
    ax = plt.axes()
    # Setting the background color
    ax.set_facecolor("white")
    plt.pcolormesh(t, f, x_train[0])
    plt.show()
'''

