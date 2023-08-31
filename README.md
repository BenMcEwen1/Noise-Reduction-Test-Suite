# Noise Reduction Test Suite

## Description
Repository accompanying paper (Understanding the Effects of Noise Reduction on Bioacoustic Segmentation and Classification
). This test suite was used to evalute segmentation and classification techniques at different Root-Mean-Squared (RMS) noise intensities. 

We include methods to generate augmented audio datasets at different RMS intensities and apply spectral subtraction for noise reduction implemented by Tim Sainbury (https://github.com/timsainb/noisereduce). This generates noisy and denoised samples for comparison.

## Dataset Generation
The raw data used to generate the results are available in `./data`. Due to the quantity of data used to generate results for each method at different RMS levels this is not included. If you want to test any of the methods outlined you can use the supplied `additive.py` script to superimpose noise and generate the 'noisy' test data for specified RMS levels and the `noise_reduction.py` script to generate the denoised dataset. Coloured noise samples used to generate RMS levels can be found at `./data/noise/`

## Segmentation
We compare three common segmentation approaches - Wavelet-based, energy and reference (STFT). The evaluation script `./segmentation/evaluate.py` will iterate through all possible methods and threshold values to identify the highest performing threshold parameter.

## Classification
We compare three pre-trained classification models (ResNet-50, Audio Spectrogram Transformer (AST) and HuBERT)fine-tuned on a binary classification task. The evaluation script `./classification/evaluate.py` will iteratively retrain each model and output performance metrics for each epoch. 

## Perceptual Metric
A subset of field recordings are used to generate perceptual quality metrics (SnNR, Success Ratio (SR), and PSNR). These results can be generated using the script `./perceptual/evaluate.py`

## Citation
Coming soon

## Acknowledgements
This research was made possible through Capability Development funding through Predator Free 2050 Limited (https://pf2050.co.nz/). AviaNZ (https://github.com/smarsland/AviaNZ) for use of their WPD method and NZ native bird dataset. Tim Sainbury (https://github.com/timsainb/noisereduce) for use of the noisereduce package (spectral subtraction)