# Federated Knowledge Distillation for Chest X-ray Classification

This project explores the integration of Federated Learning (FL) and Knowledge Distillation (KD) for multi-label chest X-ray classification using the MIMIC-CXR dataset. It supports multiple distillation strategies, adaptive loss mechanisms, and multimodal learning using both image and clinical text data.

---

## Features

* Federated Learning using Flower framework
* Multiple Knowledge Distillation strategies:

  * Static KD
  * MSE KD
  * KL Divergence KD
  * Confidence-based KD
  * Loss-adaptive KD
  * Hybrid (curriculum + confidence + adaptive)
* Multimodal learning (image + clinical text via ClinicalBERT)
* Class imbalance handling using weighted loss
* Evaluation metrics:

  * F1-score
  * Precision and Recall
  * AUPRC and AUROC
* Logging of training and evaluation metrics

---

## Project Structure

```
.
├── main_fl.py              # Entry point for FL experiments
├── fl_client.py           # Federated client logic
├── fl_server.py           # Federated server setup
├── fl_utils.py            # Dataset splitting (non-IID)

├── model.py               # Model definitions (image + multimodal)
├── dataset.py             # Dataset processing and label extraction
├── data_loader.py         # Data loading utilities
├── collate.py             # Text tokenization pipeline
├── config.py              # Configuration and experiment modes

├── train.py               # Baseline training and evaluation
├── train_static_kd.py
├── train_mse_kd.py
├── train_kl_kd.py
├── train_confidence.py
├── train_confidence_kl.py
├── train_loss_adaptive.py
├── train_loss_adaptive_kl.py
├── train_distill.py       # Hybrid distillation strategy

├── distillation.py        # Combined loss functions
├── logger.py              # Logging utilities

├── requirements.txt       # Project dependencies
└── logs/                  # Output logs
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Note:

* Ensure CUDA-compatible PyTorch is installed if using GPU.
* You may need to adjust the PyTorch install command based on your system.

---

## Dataset

Dataset: MIMIC-CXR
Format: Parquet files

The dataset contains chest X-ray images along with radiology reports. Labels are automatically extracted from reports using rule-based NLP methods.

---

## Data Loading

Data is loaded from parquet files containing images and radiology reports.

* Images are decoded and transformed using torchvision
* Reports are tokenized using a transformer tokenizer
* Labels are extracted using rule-based NLP

Custom loading logic is implemented in:

* `dataset.py`
* `data_loader.py`

---

## Model

### Image-only model

* ResNet18 or ResNet50 backbone

### Multimodal model

* Image encoder: ResNet18
* Text encoder: ClinicalBERT
* Fusion: concatenation followed by MLP

---

## Experiment Modes

Run different distillation strategies:

```bash
python main_fl.py <mode>
```

Available modes:

* baseline
* static
* mse
* kl
* loss_adaptive
* loss_adaptive_kl
* confidence
* confidence_kl
* hybrid

---

## Running Experiments

Run all modes:

```bash
python main_fl.py
```

Run a specific mode:

```bash
python main_fl.py hybrid
```

---

## Training Details

* Loss: BCEWithLogits combined with distillation loss
* Metrics:

  * F1-score
  * Precision
  * Recall
  * AUPRC
  * AUROC

---

## Federated Learning Setup

* Framework: Flower
* Strategy: FedAvg
* Data split: Non-IID using Dirichlet distribution

---

## Logging

Logs are saved in:

```
logs/<mode>_train.csv
logs/<mode>_eval.csv
```

Each log contains:

* Loss
* F1, Precision, Recall
* Lambda (distillation weight)
* Gate rate (for hybrid methods)

---

## Key Ideas

* Adaptive weighting between classification and distillation loss
* Confidence-based filtering of teacher predictions
* Curriculum learning for gradual distillation
* Hybrid strategy combining multiple approaches

---

## Future Improvements

* Add test-time evaluation
* Hyperparameter tuning
* Improve NLP-based label extraction
* Support real-world federated deployment

---

## Author

Swapnil
