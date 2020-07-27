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
model = tf.keras.models.load_model(parent_folder+"AudioDigiNet.h5")
accuracy=0
for iter in range(10):  # Iterate with mini batches 10 times
    x_test=[]
    y_test=[]
    batch_size=50
    for iter in range(batch_size):
        folder_iter = np.random.random_integers(41,60)
        which_folder = "0" + str(folder_iter)
        which_folder = which_folder[len(which_folder) - 2:len(which_folder)]
        folder = parent_folder + which_folder + "_csv/"
        list_files = sorted(os.listdir(folder))
        file_index = np.random.random_integers(500)
        file_path = folder + list_files[file_index-1]

        with open(folder + list_files[file_index-1], newline='') as csvfile:
            csv.field_size_limit(sys.maxsize)
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row = next(csvreader)
            reshaped_row = np.reshape(row[1:], (501, 101))
            x_test.append(reshaped_row[:100].astype('float32'))  # only take the first 100 frequency coefficients to reduce the size of the variables. Essentially low pass filtering for computation reduction.
            y_test.append(int(float(row[0])))

    x_test = np.array(x_test)
    x_test /= 200
    x_test = x_test.reshape(x_test.shape[0], 100, 101, 1)
    results= model.evaluate(x_test,  y_test, verbose=2)
    print(results)
    accuracy=accuracy+results[1]
accuracy=acc/10
print("Accuracy="+str(acc))
