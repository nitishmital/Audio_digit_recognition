'''
This python code takes the folder of raw audio files of different speakers speaking the digits, and saves their power spectrums as csv files for use in training the neural network.
'''

import os
import scipy.io.wavfile as wavf
import sounddevice as sd
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import csv



parent_folder="/home/nitish/PycharmProjects/AudioMNIST/data/"  # Replace with the path to the sound recordings depending on your own file system

for folder_iter in range(60):
    which_folder="0"+str(folder_iter+1)
    which_folder=which_folder[len(which_folder) - 2:len(which_folder)]
    folder=parent_folder+which_folder+"/"
    stft_folder=which_folder+"_csv/"
    if not os.path.exists(parent_folder+stft_folder):
        os.mkdir(parent_folder+stft_folder)
    list_files=sorted(os.listdir(folder))
    for i in range(len(list_files)):
        file_path = folder + list_files[i]
        label = int(list_files[i][0])
        fs, data = wavf.read(file_path)
        embedded_data = np.zeros(50000, dtype=np.int16)
        embedded_data[0:len(data)] = data
        f, t, Zxx = signal.stft(embedded_data, fs, nperseg=1000)
        power_spectrum = np.abs(Zxx)

        with open(parent_folder + stft_folder + list_files[i].split('.')[0] + ".csv", 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(np.insert(power_spectrum.flatten(), 0, label))


