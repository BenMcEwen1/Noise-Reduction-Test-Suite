import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import methods.wavelet as w
import methods.energy as e
import methods.reference as r
import sklearn.metrics as sk
import os
import shutil
from time import time
import librosa
import scipy.integrate as sp
import soundfile as sf
import csv

wave_dec_level=3
def load_truth(directory):
    t0 = time()
    with open(f'./segmentation/annotations.json') as annotations:
        dataset = json.load(annotations)
        annotations.close()
    true_binary = {}
    files = dataset.keys()
    for filename in files:
        segments = dataset[filename]
        binary = np.zeros(300)
        for segment in segments:
            start = int(np.floor(segment[0]))
            end = int(np.floor(segment[1]))
            binary[start:end] = 1
        true_binary[filename] = binary
    print(f'Annotations from {len(true_binary.keys())} files loaded in {time()-t0:.4f}s.')
    return(true_binary)

def get_correlation(data, file, wavelet, method=None):
    audio, sr = data
    t0 = time()
    if method == 'w' or method == 'wm':
        w_refs = np.load('./segmentation/evaluation/w_refs.npy', allow_pickle=True)
        data = w.wavelet_segmentation(audio, sr, w_refs, wave_dec_level)
    elif method == 'e':
        data = e.energy_segmentation(audio, sr, 7)
    elif method == 'r':
        data = r.reference_segmentation(audio, sr)
    elif method == 'p' or method == 'pm': 
        p_refs = np.load('./segmentation/evaluation/p_refs.npy', allow_pickle=True)
        data = p.parameter_segmentation(audio, sr, p_refs)
    elif method == 'm':
        data = m.mfcc_segmentation(audio, sr)
    t = time() - t0
    # print(f"{methods_names[method]} segmentation of {file} executed in {t:.4f}s.")
    # print(np.mean(data))
    return(data)

def get_binary(data, method, ratio, noise_mask):
    binary = None
    threshold = method_thresholds[method] * ratio
    if method == 'w':
        binary = w.wavelet_binary(data, threshold)
    elif method == 'e':
        binary = e.energy_binary(data, threshold)
    elif method == 'wm':
        # if np.mean(data) > 0.008:
        binary = w.dilation(np.logical_and(w.wavelet_binary(data, threshold), noise_mask), k1=1, k2=1)
        # else:
            # binary = w.wavelet_binary(data, threshold)
        # binary = w.dilation(np.logical_and(w.wavelet_binary(data, threshold), noise_mask), k1=1, k2=1)
    elif method == 'r':
        binary = r.reference_binary(data, threshold)
    elif method == 'p': 
        binary = p.parameter_binary(data, threshold)
    elif method == 'pm': 
        binary = p.dilation(np.logical_and(p.parameter_binary(data, threshold), noise_mask), k1=1, k2=1)
    elif method == 'm':
        binary = m.mfcc_binary(data, threshold)
    binary = expand_regions(binary)
    return(binary)

def noise_mask(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=2, hop_length=16000)[1,:-1]
    mfcc_binary = np.where(mfccs < max(np.mean(mfccs) - 3*np.std(mfccs), 0), 1, 0)
    noise_mask = w.dilation(mfcc_binary, k1=2, k2=2)
    return noise_mask

def expand_regions(binary):
    regions = []
    in_region = False
    start = None
    stop = None
    for i in range(len(binary)):
        if in_region:
            if binary[i]:
                continue
            else:
                stop = i-1
                in_region = False
                regions.append((start, stop))
        else:
            if binary[i]:
                start = i
                in_region = True
            else:
                continue
    if in_region:
        stop = len(binary)
        regions.append((start, stop))
    filtered_regions = []
    for region in regions:
        start, stop = region
        if stop-start <= 5:
            n = (5-(stop-start))/2
            d_start = start - n
            d_stop = stop + n
            if d_start < 0:
                d_start -= d_start
                d_stop -= d_start
            if d_stop > 300:
                d_start -= d_stop
                d_stop -= d_stop
            filtered_regions.append((int(np.floor(d_start)), int(np.floor(d_stop))))
        else:
            filtered_regions.append((int(np.floor(start)), int(np.floor(stop))))
    new_binary = np.zeros(300)
    for region in filtered_regions:
        start, stop = region
        new_binary[start:stop] = 1
    return new_binary

def get_roc(test_binary, true_binary):
    tprs = []
    fprs = []
    files = list(test_binary.keys())
    for i in range(len(test_binary[files[0]])):
        tpr = []
        fpr = []
        for file in files:
            cm = sk.confusion_matrix(true_binary[file], test_binary[file][i])
            if np.shape(cm) != (2,2):
                m = np.zeros((2,2))
                m[0,0] = cm
                cm = m
            if cm[1,1]+cm[1,0] != 0:
                tpr.append(cm[1,1] / (cm[1,1]+cm[1,0]))
            else:
                tpr.append(1)
            if cm[0,1]+cm[0,0] != 0:
                fpr.append(cm[0,1] / (cm[0,0]+cm[0,1]))
            else:
                fpr.append(1)
        tprs.append(np.mean(tpr))
        fprs.append(np.mean(fpr))
    return(tprs, fprs)

def get_statistics(test_binary, true_binary, stat_dict):
    precision = []
    recall = []
    files = list(test_binary.keys())
    for i in range(len(test_binary[files[0]])):
        confusion = np.zeros((2,2))
        for file in files:
            cm = sk.confusion_matrix(true_binary[file], test_binary[file][i])
            if np.shape(cm) != (2,2):
                m = np.zeros((2,2))
                m[0,0] = cm
                cm = m
            confusion += cm
        tn, fp, fn, tp = confusion.ravel()
        if tp + fp != 0:
            p = tp / (tp + fp)
            precision.append(p)
        else:
            p = 1
            precision.append(p)
        if tp + fn != 0:
            r = tp / (tp + fn)
            recall.append(r)
        else:
            r = 1
            recall.append(r)
        stat_dict[i]['tp'] = tp
        stat_dict[i]['tn'] = tn
        stat_dict[i]['fp'] = fp
        stat_dict[i]['fn'] = fn
        stat_dict[i]['precision'] = p
        stat_dict[i]['recall'] = r  
        stat_dict[i]['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        stat_dict[i]['overcount'] = fp / (tp + fn)
    return recall, precision, stat_dict

def evaluate(directory, methods, start, stop, step, refs):
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True)

    w_refs = np.array(w.decomp_refs(refs, wave_dec_level), dtype=object)
    # p_refs = np.array(p.decomp_refs(refs), dtype=object)

    np.save('./segmentation/evaluation/w_refs.npy', w_refs, allow_pickle=True)
    # np.save('./evaluation/p_refs.npy', p_refs, allow_pickle=True)

    true_binary = load_truth(directory)

    t0 = time()
    print('Generating audio and masks')
    audio_data = {}
    noise_masks = {}
    if os.path.exists(f'{directory}audio/'):
        for _,_,filenames in os.walk(f'{directory}audio/'):
            for filename in sorted(filenames):
                audio_data[f'{filename[:-4]}.WAV'] = np.load(f'{directory}audio/{filename[:-4]}.npy', allow_pickle=True)
                noise_masks[f'{filename[:-4]}.WAV'] = np.array([])
    else:
        # os.mkdir(f'{directory}audio/')
        for _,_,filenames in os.walk(directory):
            for filename in sorted(filenames):
                if filename.endswith('.WAV') or filename.endswith('.wav'):
                    audio, sr = librosa.load(directory+filename, sr=None)
                    if sr != 16000:
                        print(f'Resampling from {sr}Hz to 16000Hz')
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                        sf.write(f'{directory}{filename}', audio, 16000)
                    audio_data[filename] = (audio,16000)
                    # np.save(f'{directory}audio/{filename[:-4]}.npy', audio_data[filename])
                    noise_masks[filename] = np.array([])
    print(f'{len(audio_data.keys())} audio files loaded in {time()-t0:.4f}s.')
    
    for method in methods:
        t0 = time()
        test_binary = {}
        if method == 'c':
            with open(f'./classifier/classifier.json') as annotations: # Not required
                dataset = json.load(annotations)
                annotations.close()
            files = dataset.keys()
            for filename in files:
                segments = dataset[filename]
                binary = np.zeros(300)
                for segment in segments:
                    start = int(np.floor(segment[0]))
                    end = int(np.floor(segment[1]))
                    binary[start:end] = 1
                test_binary[filename] = [binary]
        else:
            for file in audio_data.keys():
                correlation = get_correlation(audio_data[file], file, wavelet, method)
                test_binary[file] = [get_binary(correlation, method, ratio, noise_masks[file]) for ratio in np.arange(start,stop,step)]

        stat_dict = [{'threshold':thr*method_thresholds[method]} for thr in np.arange(start, stop, step)]
        rec, prec, stat_dict = get_statistics(test_binary, true_binary, stat_dict)
        recall = []
        precision = []
        thresholds = []
        for i in range(len(rec)):
            if rec[i] == 'x' or prec[i] == 'x':
                continue
            else:
                recall.append(rec[i])
                precision.append(prec[i])
                thresholds.append(np.arange(start,stop,step)[i])
        recall = np.array(recall)
        precision = np.array(precision)
        data = np.concatenate((recall,precision))
        np.save(f'./segmentation/output/{method}_denoised.npy', data, allow_pickle=True)

        auc = sp.trapezoid(recall, precision)

        beta = 4

        F_beta = ((1+beta**2)*(recall*precision))/(recall + (beta**2)*precision)
        for i in range(len(stat_dict)):
            stat_dict[i][f'F_{beta}'] = F_beta[i]
        print(f"{method} segmentation evaluated in {time()-t0:.4f}s.\nAUC score: {auc}\nOptimal Cutoff: {np.arange(start,stop,step)[np.argmax(F_beta)]*method_thresholds[method]:.4f} at F_beta = {np.max(F_beta):.4f} for (recall, precision) = ({recall[np.argmax(F_beta)]:.3f},{precision[np.argmax(F_beta)]:.3f})") # for F1 of ({2*(float(recall[np.argmax(F_beta)]) * float(precision[np.argmax(F_beta)]))/(float(recall[np.argmax(F_beta)]) + float(precision[np.argmax(F_beta)]))})
        points = ([(round(recall[i], 3), round(precision[i], 3)) for i in range(len(recall))])
        # for i in range(len(points)):
        #     if points[i][0] >= 0.8 and points[i][1] > 0.05:
        #         print(f'{thresholds[i]*method_thresholds[method]:.2f}: {points[i]}')
        fields = ['threshold', 'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'accuracy', 'overcount', f'F_{beta}']
        with open(f'./segmentation/output/{method}.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(stat_dict)
            file.close()

        ax.plot(recall, precision, label=f"{methods_names[method]}", color=method_colours[method])
        ax.plot(recall[np.argmax(F_beta)],precision[np.argmax(F_beta)], color=method_colours[method], marker='o', ms=6)
    ax.legend()
    plt.show()

wavelet = 'db4'
directory = './segmentation/denoised/'

method_colours ={'w': 'tab:blue',
                'e': 'tab:purple',
                'r': 'tab:green',
                'p': 'tab:orange',
                'm': 'b',
                'wm': 'b',
                'pm': 'b',
                'c': 'b'}

ref_options = ['./segmentation/refs/ref.wav',
               './segmentation/refs/ref2.wav',
               './segmentation/refs/ref3.wav',
               './segmentation/refs/ref4.wav']

methods_names = {'w':'Wavelet Decomposition', 
                 'e': 'Spectrogram Energy', 
                 'r': 'Reference Spectrogram', 
                 'p': 'MFCC Reference',
                 'm': 'MFCC Noise Detection',
                 'wm': 'Masked Wavelet',
                 'pm': 'Masked Parameter',
                 'c': 'Classifier'}

method_thresholds = {'w': 100,
                     'e': 40,
                     'r': 160,
                     'p': 80,
                     'm': 4000,
                     'wm': 60,
                     'pm': 80,
                     'c': 1}

evaluate(directory, ['w','e','r'], 0, 1, 0.001, ref_options) #['w','e','r','p']

# with open(f'{directory}annotation.json') as annotations:
#     dataset = json.load(annotations)
#     annotations.close()

# files = list(dataset.keys())
# print(files)
# data = list(dataset.values())
# print(data)
# new = {}
# for i in range(len(files)):
#     new[files[i][7:]] = [[data[i][id]['start'],data[i][id]['end']] for id in data[i].keys()]

# print(new)

# with open(f'{directory}annotations.json', 'w') as annotations:
#     json.dump(new, annotations)
#     annotations.close()