# CNN-Based Traffic Flow Classification in 5G Core Networks

## Overview

This project explores Convolutional Neural Networks (CNNs) for classifying network traffic using two public datasets:

- **ISCX VPN-nonVPN:** Classification of VPN and Non-VPN traffic.
- **UNSW-NB15:** Classification of normal and attack traffic.

The experiments focus on dataset preprocessing, CNN training, and model evaluation.

## Project Structure

- `datasets/` — Dataset storage and processed data.
- `models/` — Saved trained CNN checkpoints.
- `notebooks/` — Jupyter notebooks.
- `reports/` — Project documentation and reports.
- `results/` — Evaluation metrics and training histories.
- `src/` — Preprocessing, training, evaluation, and validation scripts.

## Methodology

1. Inspect and preprocess each dataset separately.
2. Handle missing values and transform features as required.
3. Train CNN classifiers using the processed features.
4. Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
5. Save trained checkpoints and experimental results.

## Current Experimental Results

| Dataset | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| UNSW-NB15 | 86.19% | 80.93% | 98.01% | 88.66% |
| ISCX VPN-nonVPN | 63.88% | 60.91% | 86.02% | 71.32% |

These results come from the current saved CNN evaluations. The datasets represent different classification tasks, so their results should not be interpreted as a direct comparison of equivalent tasks.

## Current Scope

The current implementation focuses on dataset preprocessing, CNN training, and evaluation. Backend development and integration with live traffic are future work.

## Reproducibility

Install the required packages from `requirements.txt`, obtain the datasets from their authorized sources, and run the preprocessing scripts before training and evaluation.

Dataset access and redistribution must comply with the respective dataset terms.