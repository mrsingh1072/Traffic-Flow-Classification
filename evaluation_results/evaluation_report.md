# TrafficFlow - essential CNN evaluation

## Method

Fresh CPU inference using the five existing `*_100pct/model.pt` checkpoints (100% of the available training labels). Lower-label-fraction experiments are outside this five-model evaluation.

Architecture is imported unchanged from `backend/train_cnn.py`. Each checkpoint is loaded strictly, then evaluated with `model.eval()` and `torch.no_grad()`. Only `X_test`, `y_test`, and `classes` are read from the existing processed archives; no training or validation samples are used.

`datasets/preprocess_dataset.py` fits numeric median imputation and standard scaling, and categorical most-frequent imputation and one-hot encoding, on training data only. The saved test arrays already contain these transformations and are used directly: no refitting, raw-data resplitting, or double transformation. Checkpoint dataset identity, input/output dimensions, fitted preprocessing output dimension, and exact class order across checkpoint, processed data, and saved preprocessing artifact were verified.

UNSW-NB15 uses the saved deduplicated official-test partition; ISCX uses its saved grouped held-out test partitions. Metrics use the original encoded class order, with `zero_division=0` for undefined precision/recall/F1. CSV metrics are proportions (0-1).

## Results

| Dataset | Test samples | Accuracy | Macro precision | Weighted precision | Macro recall | Weighted recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UNSW_NB15 | 55,945 | 71.96% | 53.85% | 81.43% | 44.38% | 71.96% | 40.97% | 74.28% |
| ISCX_15s | 3,615 | 60.97% | 56.76% | 62.94% | 51.01% | 60.97% | 50.69% | 59.76% |
| ISCX_30s | 2,728 | 61.40% | 57.25% | 62.23% | 52.86% | 61.40% | 53.44% | 60.48% |
| ISCX_60s | 2,946 | 53.05% | 49.87% | 54.79% | 41.94% | 53.05% | 41.67% | 51.98% |
| ISCX_120s | 1,919 | 57.48% | 53.85% | 57.13% | 43.19% | 57.48% | 43.52% | 54.51% |

Highest observed macro F1: **ISCX_30s (53.44%)**. Highest observed accuracy: **UNSW_NB15 (71.96%)**.

These are different classification tasks and test distributions; the ranking does not establish that one model is universally superior.

## Weakest detection (lowest test recall)

| Dataset | Class | Recall | Precision | F1 | Support |
|---|---|---:|---:|---:|---:|
| UNSW_NB15 | Analysis | 0.90% | 3.33% | 1.41% | 446 |
| UNSW_NB15 | Backdoor | 2.02% | 29.17% | 3.78% | 346 |
| UNSW_NB15 | Worms | 6.82% | 100.00% | 12.77% | 44 |
| ISCX_15s | MAIL | 2.08% | 12.50% | 3.57% | 48 |
| ISCX_15s | STREAMING | 5.56% | 50.00% | 10.00% | 90 |
| ISCX_15s | VPN-P2P | 36.57% | 71.01% | 48.28% | 134 |
| ISCX_30s | MAIL | 0.00% | 0.00% | 0.00% | 15 |
| ISCX_30s | STREAMING | 29.03% | 62.07% | 39.56% | 62 |
| ISCX_30s | VPN-CHAT | 29.73% | 41.90% | 34.78% | 148 |
| ISCX_60s | STREAMING | 4.55% | 66.67% | 8.51% | 44 |
| ISCX_60s | VPN-MAIL | 7.69% | 26.32% | 11.90% | 65 |
| ISCX_60s | VPN-CHAT | 11.58% | 35.48% | 17.46% | 95 |
| ISCX_120s | MAIL | 0.00% | 0.00% | 0.00% | 35 |
| ISCX_120s | VPN-MAIL | 0.00% | 0.00% | 0.00% | 61 |
| ISCX_120s | STREAMING | 2.86% | 100.00% | 5.56% | 35 |

## Source artifacts and confusion matrices

| Dataset | Features | Classes | Checkpoint | Test archive | Fitted preprocessing | Matrix |
|---|---:|---:|---|---|---|---|
| UNSW_NB15 | 193 | 10 | [model.pt](../backend/models/UNSW_NB15_100pct/model.pt) | [data.npz](../datasets/processed/UNSW_NB15/data.npz) | [preprocessor.joblib](../datasets/processed/UNSW_NB15/preprocessor.joblib) | [Counts](UNSW_NB15/confusion_matrix.png) |
| ISCX_15s | 23 | 14 | [model.pt](../backend/models/ISCX_15s_100pct/model.pt) | [data.npz](../datasets/processed/ISCX_15s/data.npz) | [preprocessor.joblib](../datasets/processed/ISCX_15s/preprocessor.joblib) | [Counts](ISCX_15s/confusion_matrix.png) |
| ISCX_30s | 23 | 14 | [model.pt](../backend/models/ISCX_30s_100pct/model.pt) | [data.npz](../datasets/processed/ISCX_30s/data.npz) | [preprocessor.joblib](../datasets/processed/ISCX_30s/preprocessor.joblib) | [Counts](ISCX_30s/confusion_matrix.png) |
| ISCX_60s | 23 | 14 | [model.pt](../backend/models/ISCX_60s_100pct/model.pt) | [data.npz](../datasets/processed/ISCX_60s/data.npz) | [preprocessor.joblib](../datasets/processed/ISCX_60s/preprocessor.joblib) | [Counts](ISCX_60s/confusion_matrix.png) |
| ISCX_120s | 23 | 14 | [model.pt](../backend/models/ISCX_120s_100pct/model.pt) | [data.npz](../datasets/processed/ISCX_120s/data.npz) | [preprocessor.joblib](../datasets/processed/ISCX_120s/preprocessor.joblib) | [Counts](ISCX_120s/confusion_matrix.png) |

Confusion matrices retain the original class order. Each cell displays its exact sample count; the color scale is row-normalized so minority-class errors remain visible.

## Availability and saved outputs

All five required checkpoints, test archives, and preprocessing artifacts were available and evaluated successfully.

- `evaluation_summary.csv`: all seven requested aggregate metrics and test sample counts.
- `classification_report.csv`: per-class precision, recall, F1, support, and original class ID for every evaluated dataset.
- `<dataset>/confusion_matrix.png`: one annotated confusion matrix per dataset.

No model weights, datasets, frontend files, or backend source files were modified. No retraining or tuning was performed.
