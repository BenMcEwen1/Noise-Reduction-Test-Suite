from scipy.io import wavfile
import noisereduce as nr
import os, random
import librosa
import numpy as np

def additive_noise(original_path, noise_path, final_path):
    for _,_,filenames in os.walk(original_path):
        for filename in filenames:
            print(filename)
            noise,_ = librosa.load(noise_path, sr=16000)
            # Increase length
            scaled_noise = np.repeat(noise, 30)

            clean,rate = librosa.load(f'{original_path}{filename}', sr=16000)
            combined = np.add(clean, scaled_noise)

            wavfile.write(f"{final_path}{filename}", rate, combined)


noise_path = './brownian/brownian-5.wav'
original_path = './segmentation/dataset/'
final_path = './segmentation/extreme/original/'

additive_noise(original_path, noise_path, final_path)