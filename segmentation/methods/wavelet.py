import librosa 
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pywt
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
# import umap
import os
import soundfile as sf

count = 0

wavelet = 'dmey'
# Get the wavelet decomposition coefficients for a number of reference files
def decomp_refs(refs, wave_dec_level=3):
    ref_data = []
    for ref in refs:
        ref_audio, _ = librosa.load(ref, sr=None)
        ref_data.append(pywt.wavedec(ref_audio, wavelet, level=wave_dec_level, mode='per'))
    return ref_data


# def recursive_average(current, new, count):
#     count+=1
#     new_audio,_ = librosa.load(new, sr=None)
#     if current[0] != None:
#         mean = current + ((new_audio - current) / count)
#         return mean, count
#     else:
#         return new_audio, count

def dilation(recommend, k1=5, k2=5):
    # Expand binary mask to include surrounding areas
    if (k1 == 0) & (k2 == 0):
        return(recommend)
    d = []
    for i in range(len(recommend)):
        if i-k1 >=0:
            if np.any(recommend[i-k1:i+k2]) == 1:
                d.append(1)
            else:
                d.append(0)
        else:
            if np.any(recommend[0:i+k2]) == 1:
                d.append(1)
            else:
                d.append(0)
    return d

# For a given interval, compute and return as a list the correlation of the sampled coefficients and those in the reference
def corr_map(slice, reference):
    corrs = []
    for level in range(len(slice)):
        corrs.append(signal.correlate(slice[level], reference[level], mode='valid'))
    return(np.transpose(corrs))

def wavelet_segmentation(audio, sampleRate, ref_coeffs_list, wave_dec_level): # wavelet
    #Import and generate the coefficient matrix for the audio sample
    length_s = int(np.ceil(audio.size / sampleRate))
    coeffs_matrix = np.array(pywt.wavedec(audio, wavelet, level=wave_dec_level, mode='per'), dtype=object)
    # Get correlation value matrix
    correlations = []
    for ref_coeffs in ref_coeffs_list:
        corrs = []
        for i in range(length_s):
            slice = [coeff[i*len(coeff)//length_s:(i+1)*len(coeff)//length_s] for coeff in coeffs_matrix]
            # for s in slice:
            #     print(len(s))
            # for c in ref_coeffs[0]:
            #     print(len(c))

            corrs.append(corr_map(slice, ref_coeffs))
        correlations.append(corrs)
    correlations = np.array(correlations)

    # Sum and normalise correlation signals
    corrs_processed = (np.abs(correlations[:,:,:,1:])).sum(3)
    return(corrs_processed)

def wavelet_binary(correlations, threshold, k1=0, k2=0):
    # Compute regions
    binary_maps = np.zeros(np.shape(correlations))    
    for i in range(np.shape(correlations)[0]):
        thr = max(threshold * np.mean(correlations[i,:,:]), 0)
        binary_maps[i,:,:] = correlations[i,:,:] > thr
    binary_sum = np.array(binary_maps.sum(0) > 0)
    if k1 != 0 or k2 != 0:
        binary_sum = dilation(binary_sum, k1, k2)
    binary = np.array(binary_sum)
    return binary

# Plot
def plot_data(binary, correlations):
    fig, (ax1, ax4) = plt.subplots(nrows=2)
    # ax0.plot(np.arange(len(audio)), audio)
    print(np.shape(correlations))
    for i in np.arange(len(correlations)):
        ax1.plot(np.arange(len(correlations[i,:,0])),correlations[i,:,0])
    ax4.plot(np.arange(len(binary)), binary)
    plt.show()

ref_options = ['./audio/ref.wav',
                './audio/ref2.wav',
                './audio/ref3.wav',
                './audio/ref4.wav',]

# refs = decomp_refs(ref_options, wave_dec_level=3)

# refs = np.array(refs, dtype=object)

# audio, sr = librosa.load('/csse/users/ico29/Desktop/Sparse-Segmentation/200_dataset/20220601_220000.WAV', sr=16000)
# # audio = librosa.util.normalize(audio)
# correlations = wavelet_segmentation(audio, sr, refs, 3)
# binary = wavelet_binary(correlations, 0.08*40, 0, 0)
# # noise_mask, mfccs = mfcc_binary(audio)
# # masked_binary = dilation(np.logical_and(binary[:,0], noise_mask), k1=1, k2=1)
# print(np.count_nonzero(binary))
# plot_data(audio, binary, correlations)

# def expand(some_list, target_len):
#     multiplier = target_len//len(some_list)
#     new_list = []
#     for entry in some_list:
#         new_list.extend(multiplier*[entry])
#     return new_list


# def wavelet_svd(file):
#     audio, _ = librosa.load(file, sr=None)
#     coeffs = pywt.wavedec(audio, wavelet, level=3, mode='per')[1:]
#     new_cs = []
#     for coeff in coeffs:
#         new_c = expand(list(coeff), len(coeffs[-1]))
#         new_cs.append(new_c)
#     coeffs = np.array(new_cs)
#     svd = TruncatedSVD(n_components=2)
#     svd.fit(coeffs)
#     svs = svd.singular_values_
#     return svs

# def mfcc_svd(mfccs):
#     hop = 1600
#     samples_per_file = 80000/hop
#     pca = PCA(n_components=2)
#     points = pca.fit_transform(mfccs)
#     new = []
#     count=0
#     for point in points:
#         if count == 0:
#             temp = point
#         else:
#             temp += point
#         count+=1
#         if count == samples_per_file:
#             new.append(temp/samples_per_file)
#             count=0
#     points = np.array(new)
#     return(points)

# # def old(audio):
# #     mfcc = librosa.feature.mfcc(y=audio, n_mfcc=20)[1:]
# #     pca = PCA(n_components=2)
# #     decomp = pca.fit(mfcc)
# #     return decomp.singular_values_

# # def mfcc_umap(file):
# #     audio, _ = librosa.load(path+file, sr=None)
# #     hop = 1600
# #     samples_per_file = 80000/hop
# #     result = librosa.feature.mfcc(y=audio, n_mfcc=39, hop_length=1600).T[:-1,:]
# #     points = umap.UMAP().fit_transform(result)
# #     new = []
# #     count=0
# #     for point in points:
# #         if count == 0:
# #             temp = point
# #         else:
# #             temp += point
# #         count+=1
# #         if count == samples_per_file:
# #             new.append(temp/samples_per_file)
# #             count=0
# #     points = np.array(new)
# #     return(points)

# def kmeans_cluster(svs):
#     svs = np.array(svs)
#     km = KMeans(n_clusters=2, random_state=0).fit(svs)
#     labels = km.labels_
#     confidence = []
#     for i in range(len(labels)):
#         confidence.append(np.linalg.norm(km.cluster_centers_[labels[i]]-svs[i]))
#     confidence = np.array(confidence)
    
#     confidence = 2-(confidence - np.min(confidence))/(np.max(confidence) - np.min(confidence))
#     return confidence, labels

# def stitch_segments(directory):
#     total = np.array([])
#     filenames = read_files(directory)
#     for filename in filenames:
#         print(directory+filename)
#         audio, _ = librosa.load(directory+filename, sr=None)
#         print(np.shape(audio))
#         if np.shape(total)[0] == 0:
#             total = audio
#         else:
#             total = np.append(total, audio)
#         print(f'current length {np.shape(total)}')
#     return total

# def read_files(directory):
#     files=[]
#     for _, _, filenames in os.walk(directory):
#         for filename in filenames:
#             if filename.endswith('.WAV'):
#                 files.append(filename)
#     files.sort()
#     return files

# def save_mfccs(file, path):
#     audio, _ = librosa.load(path+file,sr=None)
#     result = librosa.feature.mfcc(y=audio, n_mfcc=39, hop_length=1600).T[:-1,:]
#     np.save(f'./mfccs/{file[:-4]}.npy',result)

# # audio = stitch_segments('./segments dataset/')
# # sf.write(f'./segments_joined.wav', audio, 16000)
# # files=read_files(path)
# # files=['segments_joined.wav']
# # points=[]
# # for file in files:
#     # points.append(mfcc_svd(file))

# # audio,_=librosa.load('./segments_joined.wav',sr=None)
# # points = mfcc_svd(audio)
# # print(points)
# # librosa.display.specshow(result.T,sr=None)

# def generate_points(directory):
#     files = read_files(directory)
#     for file in files:
#         save_mfccs(file, directory)

#     total = np.array([])
#     for file in files:
#         part = np.load(f'./mfccs/{file[:-4]}.npy')
#         if total.shape == (0,):
#             total = part
#         else:
#             total = np.append(total, part, 0)
    
#     points = mfcc_svd(total) 
#     total = normalize(total)
#     points_norm = mfcc_svd(total)
#     return(points, points_norm)

# # points, points_norm = generate_points('/csse/users/ico29/Desktop/FastAPI/backend/static/1/seg/')

# # _, labels = kmeans_cluster(points)
# # _, labels_n = kmeans_cluster(points_norm)

# # # labels = [0]

# # x = [point[0] for point in points]
# # y = [point[1] for point in points]
# # xn = [point[0] for point in points_norm]
# # yn = [point[1] for point in points_norm]
# # fig, (ax0,ax1) = plt.subplots(2)
# # ax0.scatter(x,y, np.full(len(x),20), labels, cmap="cool")
# # ax1.scatter(xn,yn, np.full(len(x),20), labels_n, cmap="cool")

# # # annotations = np.loadtxt('./segments dataset/annotations.csv',str,delimiter=',',skiprows=1,usecols=3)
# # # data = {'possum':0,'false-positive':1}
# # # true_labels = np.array([data[annotation] for annotation in annotations])

# # # scat1 = ax1.scatter(x,y, np.full(len(x),20), true_labels, cmap="cool")
# # # ax1.legend(handles=scat1.legend_elements()[0],
# # #             labels=['possum', 'false-positive'])
# # plt.show()