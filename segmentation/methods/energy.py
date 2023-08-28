import librosa 
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from time import time

# Handles rolling average and automatically accounts for padding Matrix
def moving_average(data, n):
    kernel = np.ones(n)
    data_avg = signal.convolve(data, kernel, mode='same')
    return(data_avg)

def dilation(recommend, k1=5, k2=5):
    # Expand binary mask to include surrounding areas
    d = []
    for i in range(len(recommend)):
        if i-k1 >=0:
            if any(recommend[i-k1:i+k2]) == 1:
                d.append(1)
            else:
                d.append(0)
        else:
            if any(recommend[0:i+k2]) == 1:
                d.append(1)
            else:
                d.append(0)
    return d

def energy_segmentation(audio, sampleRate, window_width):
    # Import file into usable format
    D = librosa.stft(audio)
    length_s = int(np.ceil(audio.size / sampleRate))
    # Sum each column, collapses matrix down from 1025 x n to 1 x n
    col_sums = np.sum(np.abs(D[400:,:]), axis=0, dtype=float)
    # Smooth points with a rolling average, then normalise so values are in range [0,1]
    avg_col_sums = moving_average(col_sums, window_width)
    region_size = len(avg_col_sums) // length_s
    compressed_acs = [np.max(avg_col_sums[region_size * i:region_size * (i+1)]) for i in range(length_s)]
    return(compressed_acs)

def energy_binary(avg_col_sums, threshold, k1=0, k2=0):
    threshold = threshold * np.mean(avg_col_sums)
    binary_map = [avg_col_sums[point] > threshold for point in range(len(avg_col_sums))]
    if k1 != 0 or k2 != 0:
        binary_map = dilation(binary_map, k1, k2)
    return(binary_map)

# # Plot
# def plot_data(audio, compressed_acs, binary_map, max_energies, sum_energies):
#     x = np.arange(len(binary_map))
#     fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(nrows=5)
#     ax0.plot(np.arange(len(audio)), audio)
#     ax1.plot(x, compressed_acs)
#     ax2.plot(x, binary_map)
#     ax3.bar(np.arange(len(max_energies)), max_energies)
#     ax4.bar(np.arange(len(sum_energies)), sum_energies)
#     plt.show()

# audio, sr = librosa.load('./200_dataset/20220606_003000.WAV', sr=None)
# acs = energy_segmentation(audio, sr, 7)
# binary = energy_binary(acs, 2, 4, 4)
# D = librosa.stft(y=audio, win_length=2048, hop_length=1024)
# max_energies = np.max(np.abs(D), axis=1)
# sum_energies = np.argmax(D, axis=1)
# print(np.shape(max_energies))
# plot_data(audio, acs, binary, max_energies, sum_energies)