# 👁️ Human Eye Disease Prediction
### OCT Retinal Analysis Platform — AI-powered detection of CNV · DME · DRUSEN · NORMAL

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow--CPU-2.21.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-red)
![Model](https://img.shields.io/badge/Model-MobileNetV3Large-purple)
![Dataset](https://img.shields.io/badge/Dataset-OCT%2084%2C495%20images-green)

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Disease Classes](#2-disease-classes)
3. [Dataset](#3-dataset)
4. [Project Files](#4-project-files)
5. [Environment Setup](#5-environment-setup)
6. [Model Architecture](#6-model-architecture)
7. [Training the Model](#7-training-the-model)
8. [Model Prediction](#8-model-prediction)
9. [Streamlit Web Application](#9-streamlit-web-application)
10. [Technology Stack](#10-technology-stack)
11. [Medical Disclaimer](#11-medical-disclaimer)

---

## 1. Project Overview

This project builds an AI-powered web application that classifies **Optical Coherence Tomography (OCT)** retinal scans into four categories: **CNV**, **DME**, **DRUSEN**, and **NORMAL**. It uses transfer learning with **MobileNetV3Large** pretrained on ImageNet, fine-tuned on 84,495 OCT images from Kaggle.

OCT is a non-invasive imaging technique that captures high-resolution cross-sections of the retina. Approximately **30 million OCT scans** are performed annually worldwide, and automated analysis can significantly reduce the diagnostic burden on ophthalmologists.

| Property | Details |
|----------|---------|
| Model | MobileNetV3Large (Transfer Learning) |
| Dataset | Labeled OCT — Kaggle (84,495 images) |
| Classes | CNV · DME · DRUSEN · NORMAL |
| Framework | TensorFlow-CPU 2.21.0 |
| Python | 3.12 |
| Interface | Streamlit Web App |

---

## 2. Disease Classes

| Class | Full Name | Description |
|-------|-----------|-------------|
| 🔴 **CNV** | Choroidal Neovascularization | Abnormal blood vessel growth beneath the retina. Associated with wet AMD. Visible as neovascular membrane with subretinal fluid on OCT. |
| 🟠 **DME** | Diabetic Macular Edema | Fluid accumulation in the macula due to diabetes-related vascular leakage. Appears as retinal thickening with intraretinal fluid. |
| 🟡 **DRUSEN** | Early AMD / Drusen Deposits | Yellowish deposits beneath the retinal pigment epithelium. Early marker of age-related macular degeneration. Appear as sub-RPE bumps. |
| 🟢 **NORMAL** | Healthy Retina | Preserved foveal contour with no signs of fluid, edema, or abnormal deposits. Smooth retinal layers visible on OCT. |

---

## 3. Dataset

### 3.1 Source

**Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification** — available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018).

OCT images were collected from multiple medical centres including:
- Shiley Eye Institute, University of California San Diego
- California Retinal Research Foundation
- Shanghai First People's Hospital
- Beijing Tongren Eye Center

Images were collected between **July 1, 2013 and March 1, 2017**.

### 3.2 Dataset Statistics

| Split | Images | Classes | Format |
|-------|--------|---------|--------|
| Train | 76,515 | 4 | JPEG |
| Validation | 8,000 | 4 | JPEG |
| Test | 968 | 4 | JPEG |
| **Total** | **84,495** | **4** | **JPEG** |

### 3.3 Folder Structure

```
Dataset - train+val+test/
│
├── train/
│   ├── CNV/          (37,205 images)
│   ├── DME/          (11,348 images)
│   ├── DRUSEN/       ( 8,616 images)
│   └── NORMAL/       (26,315 images)
│
├── val/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
│
├── test/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
│
├── app.py
├── recommendation.py
├── Training_model.ipynb
├── Model_Prediction.ipynb
├── Best_Model.keras
├── Trained_Model.keras
├── Trained_Model.h5
├── Training_history.pkl
└── requirements.txt
```

### 3.4 Image Labeling & Verification

Each image underwent a rigorous **three-tier grading system**:

1. **Tier 1** — Undergraduate and medical students: initial quality control, exclusion of severe artifacts
2. **Tier 2** — Four independent ophthalmologists: disease labelling for each image
3. **Tier 3** — Two senior retinal specialists (20+ years experience): final label verification

A validation subset of **993 scans** was graded separately by two ophthalmologist graders, with disagreements arbitrated by a senior retinal specialist.

---

## 4. Project Files

| File | Description |
|------|-------------|
| `Training_model.ipynb` | Full training pipeline — data loading, model building, training, evaluation |
| `Model_Prediction.ipynb` | Single image prediction with visualisation and recommendations |
| `app.py` | Streamlit web application — upload image and get prediction |
| `recommendation.py` | Medical recommendations for each disease class |
| `Trained_Model.keras` | Final trained model (Keras format) |
| `Trained_Model.h5` | Final trained model (legacy HDF5 format) |
| `Best_Model.keras` | Best checkpoint saved during training (highest val_accuracy) |
| `Training_history.pkl` | Saved training history — loss and accuracy per epoch |
| `requirements.txt` | Python package dependencies |

---

## 5. Environment Setup

### 5.1 Prerequisites

> ⚠️ **Python 3.12 is required.** TensorFlow does not have Windows wheels for Python 3.13, so Python 3.12 must be used.

Download Python 3.12 from [python.org](https://www.python.org/downloads/release/python-3120/).

### 5.2 requirements.txt

```
tensorflow-cpu==2.21.0
scikit-learn==1.8.0
numpy==2.4.4
matplotlib==3.10.1
seaborn==0.13.2
pandas==3.0.2
streamlit==1.56.0
librosa==0.11.0
```

### 5.3 Installation Steps

**Step 1** — Verify Python 3.12 is available:
```bash
py --list
```

**Step 2** — Create a virtual environment using Python 3.12:
```bash
py -3.12 -m venv tensorflow_env
```

**Step 3** — Activate the environment:
```bash
tensorflow_env\Scripts\activate
```

**Step 4** — Upgrade pip and install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 5** — Install Jupyter to run notebooks:
```bash
pip install jupyter
jupyter notebook
```

---

## 6. Model Architecture

### 6.1 Base Model — MobileNetV3Large

MobileNetV3Large is a lightweight CNN pretrained on ImageNet (1.2 million images, 1000 classes). It uses depthwise separable convolutions and hard-swish activations to achieve high accuracy with low computational cost — ideal for CPU-based inference on Windows.

### 6.2 Transfer Learning Strategy

The original ImageNet classification head is removed (`include_top=False`) and replaced with a custom head trained on the OCT dataset. The base model weights are kept trainable for **full fine-tuning**.

### 6.3 Full Model Architecture

```
Input (224 × 224 × 3)
        ↓
MobileNetV3Large  ← pretrained on ImageNet, outputs 960-dim vector
        ↓
BatchNormalization  ← normalises feature activations
        ↓
Dropout(0.3)  ← drops 30% of neurons to prevent overfitting
        ↓
Dense(256, ReLU)  ← learns disease-specific feature combinations
        ↓
Dropout(0.2)  ← additional regularisation
        ↓
Dense(4, Softmax)  ← outputs probability for each of 4 classes
        ↓
Output: [P(CNV), P(DME), P(DRUSEN), P(NORMAL)]
```

### 6.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (learning_rate = 0.0001) |
| Loss Function | Categorical Crossentropy |
| Metrics | Accuracy, F1 Score (macro average) |
| Epochs | 15 (with EarlyStopping, patience=5) |
| Batch Size | 32 (training) · 64 (test evaluation) |
| Image Size | 224 × 224 pixels |
| Input Channels | RGB (3 channels) |

### 6.5 Callbacks

| Callback | Configuration | Purpose |
|----------|--------------|---------|
| `ModelCheckpoint` | monitor=val_accuracy, save_best_only=True | Saves `Best_Model.keras` whenever val_accuracy improves |
| `EarlyStopping` | monitor=val_accuracy, patience=5 | Stops training if no improvement for 5 epochs, restores best weights |
| `ReduceLROnPlateau` | monitor=val_loss, factor=0.5, patience=3 | Halves learning rate when val_loss stalls for 3 epochs |

---

## 7. Training the Model

### 7.1 Steps

**Step 1** — Open the training notebook:
```bash
jupyter notebook Training_model.ipynb
```

**Step 2** — Update dataset paths in Cell 2:
```python
TRAIN_DIR = r'C:\path\to\your\train'
VAL_DIR   = r'C:\path\to\your\val'
TEST_DIR  = r'C:\path\to\your\test'
```

**Step 3** — Run all cells: `Cell → Run All`

**Step 4** — Files saved after training completes:

| File | Description |
|------|-------------|
| `Trained_Model.keras` | Final model after all epochs |
| `Trained_Model.h5` | Same model in legacy format |
| `Best_Model.keras` | Best checkpoint (highest val_accuracy) |
| `Training_history.pkl` | Loss and accuracy values per epoch |
| `training_curves.png` | Loss and accuracy plots |
| `confusion_matrix.png` | Confusion matrix heatmap |

### 7.2 Expected Training Time

> ⏳ Training on CPU with 76,515 images across 15 epochs typically takes **3 to 8 hours** depending on your hardware. EarlyStopping may terminate training earlier if the model converges.

---

## 8. Model Prediction

### 8.1 Using the Prediction Notebook

Open `Model_Prediction.ipynb` and update paths in Cell 2:
```python
MODEL_PATH = r'C:\path\to\Best_Model.keras'
TEST_DIR   = r'C:\path\to\test'
```

The notebook includes:
- Single image prediction with confidence scores
- Visualisation — OCT image + horizontal confidence bar chart
- Multi-class prediction — one random sample from each class
- Medical recommendation based on predicted disease
- Custom image prediction — test any image on your system

### 8.2 predict_image() Function

```python
predicted_class, confidence, all_probs = predict_image(img_path, model, CLASS_NAMES)
```

| Return Value | Type | Example |
|-------------|------|---------|
| `predicted_class` | str | `'DME'` |
| `confidence` | float | `0.91` |
| `all_probs` | dict | `{'CNV': 0.02, 'DME': 0.91, 'DRUSEN': 0.04, 'NORMAL': 0.03}` |

### 8.3 Preprocessing Pipeline

Every image goes through these steps before prediction:

1. Load and resize to **224 × 224** pixels
2. Convert to NumPy array — shape `(224, 224, 3)`
3. Add batch dimension — shape `(1, 224, 224, 3)`
4. Apply MobileNetV3 preprocessing — scales pixel values to `[-1, 1]`
5. Run `model.predict()` — returns softmax probabilities for 4 classes
6. Take `argmax` to get the predicted class index

---

## 9. Streamlit Web Application

### 9.1 Running the App

**Step 1** — Activate your virtual environment:
```bash
tensorflow_env\Scripts\activate
```

**Step 2** — Navigate to your project folder:
```bash
cd "C:\path\to\Dataset - train+val+test"
```

**Step 3** — Run the app:
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**. Press `Ctrl+C` to stop.

### 9.2 App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Platform overview, statistics, and disease information cards |
| 🔬 Disease Identification | Upload OCT image, run prediction, view confidence bars and medical recommendation |
| 📋 About | Dataset details, model architecture, and data collection methodology |

### 9.3 How to Use

1. Navigate to **Disease Identification** from the sidebar
2. Click **Browse files** and upload a JPEG or PNG OCT scan
3. Click the **Analyse Image** button
4. View the predicted disease class with confidence percentage
5. Read the confidence scores for all 4 classes
6. Scroll down to read the medical recommendation

### 9.4 Required Files

All files must be in the **same folder** as `app.py`:

```
Dataset - train+val+test/
├── app.py                ← Streamlit application
├── recommendation.py     ← Disease recommendation strings
├── Best_Model.keras      ← Trained model (primary)
└── Trained_Model.keras   ← Trained model (fallback)
```

---

## 10. Technology Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `tensorflow-cpu` | 2.21.0 | Deep learning framework — model training and inference |
| `scikit-learn` | 1.8.0 | Classification report and confusion matrix |
| `numpy` | 2.4.4 | Numerical arrays and image preprocessing |
| `matplotlib` | 3.10.1 | Training curve and prediction visualisation |
| `seaborn` | 0.13.2 | Confusion matrix heatmap styling |
| `pandas` | 3.0.2 | Data manipulation and analysis |
| `streamlit` | 1.56.0 | Interactive web application interface |
| `librosa` | 0.11.0 | Audio processing (available for future extensions) |

---

## 11. Medical Disclaimer

> ⚠️ **IMPORTANT:** This tool is intended for **educational and research purposes only**. It is **NOT** a substitute for professional medical diagnosis, clinical evaluation, or the advice of a qualified ophthalmologist. Always consult a licensed medical professional for any health concerns or before making any medical decisions based on this tool's output.

---

*Human Eye Disease Prediction — OCT Retinal Analysis Platform*
