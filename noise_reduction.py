from scipy.io import wavfile
import noisereduce as nr
import os
import librosa

NOISY_PATH = './segmentation/dataset/'
CLEAN_PATH = './segmentation/denoised/'

def denoise():
    for _,_,filenames in os.walk(NOISY_PATH):
        for filename in sorted(filenames):
            print(filename)
            data, rate = librosa.load(f"{NOISY_PATH}{filename}", sr=16000)
            reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
            wavfile.write(f"{CLEAN_PATH}{filename}", rate, reduced_noise)

denoise()