import librosa 
import librosa.display
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from time import time

def extract():
    # Extract recordings, replace with SD card directory
    masks = []
    PATH = "./segmentation/reference/"
    for _, _, filenames in os.walk(PATH):
        for filename in filenames:
            if filename == "possum_snip4.npy":
                data = np.load(f"{PATH}{filename}")
                masks.append(data)
    return masks

def spectrograms(recording, sampleRate, plot=False):
    # Plot spectrograms of recording and ref
    _, _, Sp = signal.spectrogram(recording, fs=sampleRate)
    return Sp

def normalise(mask):
    # Normalise to prevent higher energy masks becoming biased
    norm = np.linalg.norm(mask)
    mask = np.divide(mask, norm)
    mask = mask / mask.sum()
    return mask

def correlation(recording, masks, sampleRate):

    # Convolve spectrogram with ref to generate correlation
    Sp = spectrograms(recording, sampleRate)

    # Normalisation
    Sp = normalise(Sp)
    kernel = np.ones((2,2)) * 0.5
 
    cor = []
    scaled = []

    lower = 0
    upper = 0.0033

    for mask in masks:
        # Normalise Mask
        mask = normalise(mask)
        mask = signal.convolve2d(mask, kernel, mode='same', boundary='wrap', fillvalue=0)
        c = signal.correlate(Sp, mask, mode="valid")
        cor.append(c[0])
    
    for c in cor:
        c = np.interp(c, (lower,upper), (0,10)) 
        scaled.append(c)

    return cor

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

def reference_segmentation(audio, sampleRate):
# Extract reference and audio file
    mask = extract()
    length_s = int(np.ceil(audio.size / sampleRate)) 
    # Compute correlation values by convolving reference mask across audio sample
    cor = correlation(audio, mask, sampleRate)
    # Normalise correlation so values are in range [0,1]
    corr = cor[0]/np.max(cor[0])
    # Find regions using the correlation between the mask and the sample
    chunk_size = len(corr) // length_s
    corr_compressed = np.array([max(corr[chunk_size * i:chunk_size * (i+1)]) for i in range(length_s)])
    return(corr_compressed)

def reference_binary(corr, threshold, k1=0, k2=0):
    binary_map = np.zeros(np.shape(corr))
    threshold = threshold * np.mean(corr)
    binary_map[:] = corr[:] > threshold
    if k1 != 0 or k2 != 0:
        binary_map = dilation(binary_map, k1, k2)
    # plot(corr, binary_map)
    return(binary_map)

# Plot
def plot(cor_norm, compressed_regions):
    x = np.arange(cor_norm.size)
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.plot(x, cor_norm)
    ax1.plot(np.arange(len(compressed_regions)), compressed_regions)
    plt.show()

# audio, sr = librosa.load('/csse/users/ico29/Desktop/Sparse-Segmentation/test_dataset/20220806_001000.WAV', sr=16000)
# reference_binary(reference_segmentation(audio, sr), 1)