# Stronger CNN training results

Existing serving checkpoints, application code, datasets, preprocessing and class mappings remain unchanged. All candidates retain the original TrafficCNN-1D architecture and state-dictionary keys.

Training used existing training splits only. Validation selected starts, regularized fine-tuning checkpoints, and a small global class-prior correction. UNSW-NB15 used frozen convolutional embeddings with head fine-tuning; ISCX used full-model fine-tuning. Each training arm used natural or square-root-inverse-frequency weighted cross entropy, 0.02 label smoothing, AdamW with weight decay 0.001, validation-controlled learning-rate reduction and early stopping (maximum 50 epochs, patience 8). The prior correction uses training class frequencies and is absorbed into the existing final Linear bias.

All checkpoint and prior-correction selection used validation macro F1, with validation accuracy no worse than one percentage point below the original. The selected checkpoint for each dataset was assessed once on its existing test split; test results did not select epochs or correction strength. Previous development already examined these test sets, so this comparison is not independent confirmation on new data.

| Dataset | Original accuracy | Candidate accuracy | Original macro F1 | Candidate macro F1 | Status |
|---|---:|---:|---:|---:|---|
| UNSW_NB15 | 71.96% | 66.78% | 40.97% | 43.49% | test_did_not_confirm_keep_original |
| ISCX_15s | 60.97% | 61.77% | 50.69% | 56.98% | improved_candidate_not_deployed |
| ISCX_30s | 61.40% | 63.38% | 53.44% | 54.69% | improved_candidate_not_deployed |
| ISCX_60s | 53.05% | 55.06% | 41.67% | 42.78% | improved_candidate_not_deployed |
| ISCX_120s | 57.48% | 58.42% | 43.52% | 47.44% | improved_candidate_not_deployed |

Retained outputs: `comparison.csv`, `classification_report.csv`, and the four ISCX directories containing `model.pt`, `metrics.json` and `confusion_matrix.csv`. The rejected UNSW-NB15 candidate and generated training traces were removed during repository cleanup. The original UNSW-NB15 checkpoint remains in `backend/models/UNSW_NB15_100pct/model.pt`. These ISCX candidates are not deployed. Original checkpoint SHA-256 values were verified unchanged.

## Minority-class changes and remaining limits

- ISCX 15s MAIL recall improved from 2.08% to 31.25% (48 test samples); STREAMING recall improved from 5.56% to 54.44% (90 samples).
- ISCX 120s VPN-MAIL recall improved from 0% to 45.90% (61 samples).
- ISCX 30s MAIL still has 0% recall (15 samples).
- ISCX 60s has an accuracy/macro-F1 gain over the original, but STREAMING recall fell from 4.55% to 2.27%. The previous fine-tuned candidate has higher test macro F1 (43.73% vs 42.78%) but lower accuracy (52.92% vs 55.06%). Neither dominates the other. The earlier checkpoint remains only as a reproducibility input for the final training run.
- The UNSW-NB15 candidate lost too much test accuracy (71.96% to 66.78%); keep the original model.
- Improvements do not make these models strong production detectors. Rare-class weaknesses and limited held-out generalization remain.

## Reproducibility inputs

`train_stronger.py` uses the baseline summary at `../evaluation_summary.csv`, and the warm-start checkpoints and comparison table in `../improvement/`. Those inputs and `../improvement/train_candidates.py` are intentionally retained. They are training dependencies, not additional serving models. Fitted preprocessors and processed datasets remain at their existing project locations.
