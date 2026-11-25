# Week 1: Core Deep Learning, Transformers & Inference Foundations

## Overview

This repository covers the foundational engineering track for building, fine-tuning, exporting, and running inference on neural network and Transformer-based NLP models.

It progresses from core PyTorch mechanics and custom training loops to Transformer architectures, LoRA fine-tuning, Hugging Face Hub distribution, error analysis, and production-minded inference pipelines.

## Learning Objectives

By completing this week, the following engineering concepts and practical skills were covered:

- **PyTorch & Autograd Mechanics:** Working directly with tensors, automatic differentiation, and custom training loops without relying exclusively on high-level abstractions.
- **Custom Modules & Mixed Precision (AMP):** Designing custom `nn.Module` components, Dataset/DataLoader pipelines, and mixed-precision training to reduce memory usage and improve execution efficiency.
- **Hugging Face Ecosystem:** Managing tokenization, padding, truncation, attention masks, context-length constraints, and AutoModel workflows.
- **Transformer Internals:** Understanding attention mechanisms, normalization, feed-forward networks, and structural differences between encoder, decoder, and encoder-decoder architectures.
- **Parameter-Efficient Fine-Tuning (LoRA):** Comparing full fine-tuning against LoRA/PEFT for sequence classification and evaluating trade-offs involving rank (`r`), learning rate, trainable parameters, and training time.
- **Hub Export, Error Analysis & Inference:** Exporting models and adapters, documenting model cards and failure modes, and separating inference pipelines from training scripts.

---

## Repository Structure

```text
week1_core/
├── README.md
├── day1_pytorch_autograd.py
├── day2_custom_layers_Amp.py
├── day3_HuggingFace.py
├── day4_Transformer.py
├── day4_Transformer_Pro_Mini_model.py
├── day5_Fine-Tuning_Vs_LoRA_Project.py
└── day6-hf-hub-export/
    ├── hf_hub_export.py
    ├── inference.py
    ├── model_card.md
    ├── error_analysis.md
    └── README.md
```

---

## Experiments & Results

### Fine-Tuning Comparison

A comparative experiment was conducted using full fine-tuning and LoRA on the same base model and dataset.

**Base Model:** `distilbert-base-uncased`

**Dataset:** `nyu-mll/glue` (`sst2`)

**Evaluation Split:** `validation` (872 samples)

### Training Configuration

| Parameter | Value |
| :--- | :--- |
| Maximum Sequence Length | `128` |
| Batch Size | `16` |
| Epochs | `3` |
| Random Seed | `42` |
| Full Fine-Tuning Learning Rate | `2e-5` |
| LoRA Learning Rate | `2e-4` |
| Weight Decay | `0.01` |

### Performance Comparison

| Metric | Full Fine-Tuning | LoRA (r = 16) |
| :--- | :---: | :---: |
| Trainable Parameters | `66,955,010` | `887,042` |
| Trainable Share | `100%` | `~1.32%` |
| Accuracy | `0.9197` | `0.9002` |
| Macro F1 | `0.9197` | `0.9001` |
| Training Time | `629.38s` | `452.24s` |

### Confusion Matrices

**Full Fine-Tuning**

| | Predicted Negative | Predicted Positive |
| :--- | :---: | :---: |
| **Actual Negative** | 389 | 39 |
| **Actual Positive** | 31 | 413 |

**LoRA (r = 16)**

| | Predicted Negative | Predicted Positive |
| :--- | :---: | :---: |
| **Actual Negative** | 378 | 50 |
| **Actual Positive** | 37 | 407 |

> **Note:** All reported evaluation metrics are based on the SST-2 validation split, not the test split.

---

## Hugging Face Hub Models

Both trained models were exported and published to Hugging Face Hub.

### Full Fine-Tuned Model

**Repository:** [`MohamedSayed112/sst2-distilbert-full-ft`](https://huggingface.co/MohamedSayed112/sst2-distilbert-full-ft)

Contains the complete fine-tuned model weights and configuration required for standalone loading.

### LoRA Adapter

**Repository:** [`MohamedSayed112/sst2-distilbert-LoRA`](https://huggingface.co/MohamedSayed112/sst2-distilbert-LoRA)

Contains the LoRA adapter weights and configuration. The original base model must be loaded before attaching the adapter.

---

## Error Analysis

The validation predictions were analyzed using confusion matrices and classification metrics to identify incorrect predictions and common failure patterns.

### Misclassification Summary

| Model | Total Errors | False Positives | False Negatives |
| :--- | :---: | :---: | :---: |
| Full Fine-Tuning | `70 / 872` | `39` | `31` |
| LoRA (r = 16) | `87 / 872` | `50` | `37` |

### Common Failure Modes

The observed error patterns included:

- **Sarcasm and Contrast Clauses:** Sentences where individual words suggest a different sentiment from the overall meaning.
- **Short or Ambiguous Expressions:** SST-2 phrases that provide insufficient context for reliable sentiment classification.
- **Sequence Truncation:** Potential information loss when input sequences exceed `max_length = 128`.

---

## Engineering Decisions & Trade-Offs

### 1. Parameter Efficiency vs. Peak Performance

LoRA (`r = 16`) updates only `887,042` parameters, approximately `1.32%` of the model's total parameters.

It achieved a shorter training time (`452.24s` compared with `629.38s` for full fine-tuning), while full fine-tuning achieved higher validation accuracy (`0.9197` vs. `0.9002`) and fewer validation errors.

### 2. Separation of Training & Inference

Training scripts are kept separate from model-export and inference pipelines.

This separation allows inference to run independently without requiring the training workflow or optimizer-related operations.

### 3. Safe Inference Execution

The inference implementation uses:

- `model.eval()` to switch the model to evaluation mode.
- `torch.inference_mode()` to disable gradient tracking during prediction.
- Explicit device placement (`cuda` when available, otherwise `cpu`).
- Batch processing to support multiple input samples.
- Input validation to reject empty or non-string inputs.

---

## Quick Usage

### 1. Install Dependencies

```bash
pip install torch transformers peft huggingface_hub
```

### 2. Export Models to Hugging Face Hub

Authenticate with Hugging Face before uploading.

```bash
hf auth login
```

Review the local model paths and repository IDs inside:

```text
week1_core/day6-hf-hub-export/hf_hub_export.py
```

Then execute:

```bash
python week1_core/day6-hf-hub-export/hf_hub_export.py
```

> **Security Note:** Never hardcode or commit Hugging Face access tokens to a public repository. Use secure authentication or environment variables.

### 3. Run Inference

Open `inference.py` and select the desired model type:

```python
MODEL_TYPE = "full"
```

Or:

```python
MODEL_TYPE = "lora"
```

Then execute:

```bash
python week1_core/day6-hf-hub-export/inference.py
```

---

## Key Outcomes

By the end of Week 1, the repository demonstrates practical experience with:

- Building custom PyTorch modules and training loops.
- Understanding Transformer architecture and attention mechanisms.
- Fine-tuning pretrained models using full fine-tuning and LoRA.
- Comparing model performance and parameter efficiency.
- Exporting and publishing trained models to Hugging Face Hub.
- Performing validation-based error analysis.
- Building reusable inference pipelines with batch processing and safe execution practices.

---

**Week 1 Focus:** Establishing strong Deep Learning and Transformer engineering foundations, with practical model fine-tuning, deployment preparation, and reproducible inference workflows.