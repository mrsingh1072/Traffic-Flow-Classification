# CNN-Based Traffic Flow Classification in 5G Core Networks

## Overview

This project explores Convolutional Neural Networks (CNNs) for network traffic classification using two public network traffic datasets: **UNSW-NB15** and **ISCX VPN-nonVPN**.

The system focuses on preprocessing network traffic features, training CNN-based classifiers, evaluating model performance, and providing predictions through a FastAPI backend.

The project also investigates how the amount of labeled training data affects classification performance by training models with different label fractions.

### Datasets

- **UNSW-NB15:** Classification of normal network traffic and multiple attack categories.
- **ISCX VPN-nonVPN:** Classification of VPN and non-VPN network traffic using flow-based features.

The ISCX dataset experiments use four traffic-flow configurations: 15-second, 30-second, 60-second, and 120-second flows.

## Project Objectives

- Prepare public network traffic datasets for machine learning.
- Develop a consistent preprocessing pipeline for each dataset.
- Train 1D CNN models for traffic classification.
- Evaluate classification performance using standard machine learning metrics.
- Investigate the effect of training-label availability on model performance.
- Develop a FastAPI backend for traffic classification.
- Provide prediction results, confidence scores, and evaluation metrics through API endpoints.
- Provide a schema-driven frontend for classification and model evaluation.

## Project Structure

```text
Traffic-Flow-Classification/
├── backend/
│   ├── main.py
│   ├── train_cnn.py
│   ├── evaluate_model.py
│   ├── evaluate_all_datasets.py
│   ├── visualize_results.py
│   ├── test_api.py
│   ├── test_all_datasets.py
│   ├── test_api_validation.py
│   ├── test_backend_ready.py
│   ├── test_batch.py
│   ├── models/
│   └── results/
├── datasets/
│   ├── CSV Files/
│   ├── processed/
│   ├── preprocess_dataset.py
│   └── verify_dataset.py
├── requirements.txt
├── README.md
└── venv/
```

*The structure above reflects the files and folders identified during development. Additional folders and files may exist locally.*

## Methodology

### Phase 1: Dataset Preparation

1. Obtain the public datasets from their authorized sources.
2. Inspect dataset formats, features, labels, and class distributions.
3. Handle missing values and categorical features.
4. Remove or handle duplicate records where appropriate.
5. Fit preprocessing pipelines separately for each dataset.
6. Transform traffic features into numerical representations suitable for CNN input.
7. Save processed datasets and preprocessing artifacts.
8. Prepare training, validation, and test splits.
9. Verify processed data and check for exact overlap between splits.

Processed datasets are stored under `datasets/processed/`. Each dataset has a saved `data.npz` file and a `preprocessor.joblib` artifact.

### Phase 2: CNN Model Development

A one-dimensional CNN (1D CNN) is used to classify traffic-flow feature vectors.

The architecture includes:

- 1D convolutional layers
- Batch normalization
- ReLU activation functions
- Max pooling
- Adaptive max pooling
- Fully connected layers
- Dropout for regularization
- An output layer corresponding to the dataset's classes

Separate models and preprocessing artifacts are maintained for the different dataset configurations.

### Label-Fraction Experiments

To investigate the impact of labeled training data, models were trained using four label fractions:

- 10%
- 25%
- 50%
- 100%

The experiments cover five dataset configurations, resulting in 20 training configurations.

### Phase 3: Model Evaluation

Models are evaluated using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Confusion matrices
- Per-class precision, recall, and F1-score

Evaluation scripts save results under `backend/results/`, including dataset-level evaluation summaries and visualization artifacts.

## Current Experimental Results

The following results are from the processed held-out test splits evaluated during development.

| Dataset | Test samples | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| UNSW-NB15 | 55,945 | 71.96% | 53.85% | 44.38% | 40.97% | 74.28% |
| ISCX-15s | 3,615 | 60.97% | 56.76% | 51.01% | 50.69% | 59.76% |
| ISCX-30s | 2,728 | 61.40% | 57.25% | 52.86% | 53.44% | 60.48% |
| ISCX-60s | 2,946 | 53.05% | 49.87% | 41.94% | 41.67% | 51.98% |
| ISCX-120s | 1,919 | 57.48% | 53.85% | 43.19% | 43.52% | 54.51% |

These metrics describe different dataset configurations and classification tasks. They should not be interpreted as a direct comparison of equivalent classification problems.

**Performance note:** Macro F1 is lower than weighted F1 for several configurations, indicating that performance varies across classes. Further model analysis and improvements may be needed, particularly for minority traffic classes.

## Phase 4: FastAPI Backend

A FastAPI backend has been implemented to expose the trained models and their evaluation results.

### Implemented Capabilities

- Health checking
- Dataset and model availability
- Dataset-specific feature schemas
- Single-record traffic prediction
- Batch prediction
- Overall evaluation metrics
- Per-class evaluation metrics
- Confusion-matrix data
- Input validation and invalid-dataset handling

The prediction pipeline accepts raw traffic features, applies the corresponding saved preprocessing pipeline, and passes the resulting features to the CNN model.

Prediction responses include the predicted class, class identifier, confidence score, and class-wise scores where supported.

**Confidence note:** The returned confidence is a softmax score and should not be interpreted as a calibrated probability.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Backend health status |
| GET | `/datasets` | Available dataset configurations |
| GET | `/schema` | Default feature schema |
| POST | `/predict` | Single-record prediction |
| POST | `/predict/batch` | Batch traffic prediction |
| GET | `/metrics` | Overall evaluation metrics |
| GET | `/metrics/{dataset}/classes` | Per-class evaluation metrics |
| GET | `/metrics/{dataset}/confusion-matrix` | Confusion-matrix data |

Dataset-specific schemas are available through the schema endpoint using its supported query parameters.

### API Testing

The backend readiness test passed **15 out of 15 checks** during development.

Verified checks included:

- Health endpoint response
- Model and preprocessing artifact availability
- Dashboard metrics
- Schema and prediction for all five dataset configurations
- Rejection of invalid dataset names

Additional testing confirmed successful batch prediction and successful responses from the per-class metrics and confusion-matrix endpoints for UNSW-NB15. The Python dependency consistency check also passed.

These results verify the tested backend functionality; they do not constitute production deployment or security certification.

## Installation and Setup

### Requirements

- Python 3.11
- FastAPI
- Uvicorn
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Joblib

The complete dependency list is maintained in `requirements.txt`.

### 1. Clone or Obtain the Project

Obtain the project files and place the datasets in their expected locations. Dataset access and redistribution must comply with the respective dataset terms.

### 2. Create and Activate the Virtual Environment

On Windows:

```cmd
cd /d D:\Traffic-Flow-Classification
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```cmd
python -m pip install -r requirements.txt
python -m pip check
```

### 4. Run the Backend

From the project root:

```cmd
cd backend
python -m uvicorn main:app --reload
```

The API will normally be available at:

- **API base URL:** http://127.0.0.1:8000
- **Swagger documentation:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health
- **Available datasets:** http://127.0.0.1:8000/datasets
- **Evaluation metrics:** http://127.0.0.1:8000/metrics

Keep the terminal running while using the API.

## Running Tests

With the virtual environment activated and the API server running, open another terminal at the project root.

Run the backend readiness test:

```cmd
python backend\test_backend_ready.py
```

Run API validation tests:

```cmd
python backend\test_api_validation.py
```

Run dataset smoke tests:

```cmd
python backend\test_all_datasets.py
```

Check Python dependency consistency:

```cmd
python -m pip check
```

The batch prediction test can also be run using:

```cmd
python backend\test_batch.py
```

## Phase 5: Frontend Application

The existing React frontend now includes Overview, Traffic Classifier, Model Evaluation, Dataset Explorer, and About workspaces in the Black Signal dark design system with Space Grotesk headings, Inter body, and horizontal desktop navigation. Single and batch predictions use the original FastAPI endpoints and dataset-specific schemas; charts and tables use returned evaluation data. A custom interactive SVG logical globe illustrates 5G components, with a node inspector, orbit control, layer visibility, and user-triggered packet traces, and layouts support desktop and mobile devices.

See [frontend setup and verification](frontend/README.md) for complete instructions. With the backend running, start the frontend in a second terminal:

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Set `VITE_API_BASE_URL` using `frontend/.env.example` if the API is hosted elsewhere. `npm run build` runs strict TypeScript checks and creates the production bundle. All five canonical hash routes support reload and browser history. Local validation passed 5 route regression tests, 15 readiness checks, 8 API validation checks, and 48 cross-dataset contract checks. Browser checks covered prediction, batch result selection, evaluation, mobile layouts, and navigation.

## Limitations and Future Work

- Classification performance varies by dataset and traffic class.
- Minority classes require further investigation and potentially improved training strategies.
- Softmax confidence scores are not calibrated probabilities.
- The current prediction API classifies supplied feature records; it does not directly capture live network traffic.
- The frontend requires a reachable backend with the existing model and preprocessing artifacts.
- Production deployment, authentication, security hardening, and monitoring require separate work.

Potential future improvements include class-imbalance handling, hyperparameter tuning, calibration analysis, additional evaluation, and integration with a live traffic-capture pipeline.

## Reproducibility

For reproducible experiments:

1. Use the documented Python environment and dependency versions.
2. Obtain the datasets from their authorized sources.
3. Run preprocessing before training.
4. Retain the fitted preprocessing artifacts alongside the corresponding model checkpoints.
5. Use the provided training and evaluation scripts.
6. Preserve dataset splits and saved evaluation results for comparison.

Dataset access and redistribution must comply with the respective dataset terms.

## Project Status

- **Dataset preparation:** Completed
- **CNN training:** Completed for the current experiment configurations
- **Model evaluation:** Completed for the reported test splits
- **FastAPI backend:** Core functionality implemented and tested
- **Frontend:** Implemented; strict TypeScript production build verified
- **End-to-end integration:** Verified locally against all five dataset configurations
- **Production deployment:** Pending
