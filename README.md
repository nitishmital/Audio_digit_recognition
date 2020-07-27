# Audio_digit_recognition

# Data - https://github.com/soerenab/AudioMNIST.git
They provide recordings of all 10 digits spoken by multiple people with different accents. The recordings are in the folder data in their repository. Also, since they request me to cite their paper if I use their data in my project, here goes - https://arxiv.org/abs/1807.03418

I use a simple educational approach. I convert the raw wav files to a spectrogram using Short Time Fourier Transform. Then, the lower frequency components of the STFT are viewed a images, and a convolutional neural network is trained on it. The architecture consists of 1 conv layer, 1 max pooling layer, and 2 fully connected layers.
