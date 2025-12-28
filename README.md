# Automated Structural Defect Detection

An end-to-end deep learning system for **automated concrete crack detection and weak localization**, built with a strong focus on **generalization, explainability, and engineering rigor**.

This project goes beyond basic image classification to address real-world challenges in visual inspection systems, including dataset bias, evaluation discipline, and model interpretability using explainable AI techniques.

**Tech Stack:**
Python · PyTorch · Torchvision · NumPy · Scikit-learn · OpenCV · Matplotlib

---

## Problem Statement

Manual inspection of concrete structures is:

* Time-consuming and labor-intensive
* Subjective and inconsistent across inspectors
* Difficult to scale across large infrastructure assets

Automated visual inspection systems aim to address these limitations by leveraging computer vision models that can:

* Reliably detect surface-level defects such as cracks
* Generalize across varying textures and lighting conditions
* Provide interpretable predictions suitable for engineering review

The goal of this project is to design a **robust and explainable computer vision pipeline** for automated concrete crack detection using real-world inspection data.

---

## Key Features

* End-to-end deep learning pipeline (data loading → training → evaluation)
* Binary crack detection using transfer learning (ResNet50)
* Proper train / validation / test separation
* Robust evaluation using Precision, Recall, F1-score, and ROC-AUC
* Weakly supervised crack localization using Grad-CAM
* Qualitative error inspection and model explainability
* Fully reproducible training via a single end-to-end notebook

---

## System Overview

```
Raw Concrete Images (SDNET2018)
        |
        v
Data Loading & Augmentation
        |
        v
CNN Training (Transfer Learning)
        |
        v
Validation-Based Model Selection
        |
        v
Test Set Evaluation (F1, ROC-AUC)
        |
        v
Grad-CAM Explainability & Weak Localization
```

---

## Model Architecture

| Component         | Description                            |
| ----------------- | -------------------------------------- |
| Backbone          | ResNet50 (ImageNet pretrained)         |
| Training Strategy | Transfer learning with frozen backbone |
| Output            | 2-class softmax (crack / no crack)     |
| Loss Function     | Cross-Entropy Loss                     |
| Optimizer         | AdamW                                  |

ResNet50 was selected due to its strong performance on texture- and edge-driven visual tasks, which are critical for crack detection.

---

<img width="1174" height="407" alt="image" src="https://github.com/user-attachments/assets/16bdd822-a367-48bd-9a9e-432e33f363fe" />

## Explainability with Grad-CAM

The SDNET2018 dataset provides **image-level labels only** and does not include bounding-box annotations.

To enable localization and interpretability, the project applies **Grad-CAM (Gradient-weighted Class Activation Mapping)** to:

* Highlight regions contributing most to crack predictions
* Verify that the model focuses on crack patterns rather than background texture
* Support human-in-the-loop inspection workflows

This enables **weak localization** of cracks without explicit spatial supervision.

---

## Dataset

<img width="1470" height="594" alt="image" src="https://github.com/user-attachments/assets/dd6b96e3-f520-42fa-bcf2-2abd1bb7de8e" />

### Dataset Source

The dataset used in this project is **SDNET2018**, a publicly available benchmark dataset for concrete crack detection.

Due to licensing and size constraints, the dataset is **not included** in this repository.

### Expected Directory Structure

```
data/
└── SDNET/
    ├── train/
    │   ├── NEGATIVE/
    │   └── POSITIVE/
    ├── val/
    │   ├── NEGATIVE/
    │   └── POSITIVE/
    └── test/
        ├── NEGATIVE/
        └── POSITIVE/
```

Users are expected to download the dataset separately and place it under the `data/` directory.

---

## Project Structure

```
automated-structural-defect-detection/
├── src/                # Core training, evaluation, and explainability code
├── notebooks/          # End-to-end reproducible notebook
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shreyas-Gaikwad/automated-structural-defect-detection.git
cd automated-structural-defect-detection
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

---

## Usage

### Run End-to-End Notebook

The primary entry point for this project is the end-to-end notebook:

```
notebooks/crackDetectionCNN.ipynb
```

The notebook:

* Trains the model from scratch (transfer learning)
* Evaluates on validation and test sets
* Visualizes Grad-CAM crack localization

No pretrained weights are required.

---

## Contributing

Contributions are welcome.

Please open issues or submit pull requests for:

* Bug fixes
* Documentation improvements
* Experimental extensions

---

## Author

Built by **Shreyas Gaikwad**
Focus: Computer Vision · Deep Learning · Automated Visual Inspection
