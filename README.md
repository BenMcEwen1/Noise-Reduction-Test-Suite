# Noise Reduction Test Suite
Paper - Understanding the Effects of Noise Reduction on Audio Segmentation and Classification

## Data
Brownian noise samples can be found in `./brownian/brownian{RME}.wav` with separate files for each RMS noise level.  `./brownian` also includes the evaluation dataset used to generate SnNR, SR and PSNR results at each RMS noise level.

### Dataset Generation
The raw data used to generate the results are available in `./data`. Due to the quantity of data used to generate results for each method at different RMS levels e.g. `26 * data` this is not included. If you want to test any of the methods outlined you can use the supplied `additive.py` script to superimpose noise and generate the 'noisy' test data for specified RMS levels and the `noise_reduction.py` script to generate the denoised dataset.