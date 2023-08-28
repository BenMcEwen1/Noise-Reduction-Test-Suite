# Classification

Evaluate the performance of three common classifications model ResNet-50, AST and HuBERT on unprocessed and denoised data. 

Training script `training.py` retrains each model using the unprocessed and denoised data and outputs the results for each.

Different datasets are used for segmentation and classification because the segmentation is operating on the full 5-minute field recording whereas the classifiers operate on 5-second segments. The original field data is the same.