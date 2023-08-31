from scipy.io import wavfile
import noisereduce as nr
import os, random
import librosa
import numpy as np
import pandas as pd

# annotations_file = './classification/mini.csv'
# annotations = pd.read_csv(annotations_file)
# files = annotations.iloc[:,1].to_list()

def additive_noise(original_path, noise_path, final_path):
    for _,_,filenames in os.walk(original_path):
        for filename in filenames:
            # Check if filename in annotation.csv
            # if filename in files:
            print(filename)
            noise,_ = librosa.load(noise_path, sr=16000)

            # scaled_noise = np.repeat(noise, 30) # For 5-min recordings
            # scaled_noise = noise[0:80000] # For 5-second segments

            try:
                clean,rate = librosa.load(f'{original_path}{filename}', sr=16000)
                combined = np.add(clean, noise)
                wavfile.write(f"{final_path}{filename}", rate, combined)
            except:
                print(f"Error for {filename}")


noise_path = './data/noise/white/white-10.wav'
original_path = './predator/'
final_path = './data/noise/white/data/extreme/original/'

additive_noise(original_path, noise_path, final_path)