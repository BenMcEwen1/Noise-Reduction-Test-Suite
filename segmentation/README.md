# Segmentation Evaluation

## Dataset
Contains 200 5-minute field recordings. `annotations.json` contains the ground truth segmentation results

## Methods
Each method can be found in `/methods`. The evaluation file `evaluate.py` runs each of the selected methods.

Note - this may take some time. The evaluation method iterates through each threshold parameter to identify the optimal cutoff