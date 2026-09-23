# Coral Reef Health Classification 🪸

A deep learning model that looks at a photo of coral and tells you if it's **healthy or bleached**.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

---

## What it does

Coral bleaching is one of the clearest signs of reef stress from warming oceans, and right now checking reef health mostly means divers manually inspecting and photographing coral. This project automates the first step: given an underwater photo, it classifies the coral as **healthy** or **bleached**.

Instead of relying on just one model, it combines the opinions of two:
1. A **custom CNN** (a neural network built from scratch for this task) with an added "attention" mechanism that helps it focus on the most relevant parts of the image, instead of the whole picture equally.
2. **VGG19**, a well-known image-recognition model that's already been trained on millions of general images, fine-tuned here specifically for coral photos.

Their predictions are then averaged together — an **ensemble** — which performs better than either model alone.

## Results

| Model | Test Accuracy |
|-------|---------------|
| Custom CNN (with attention) | ~80% |
| VGG19 (fine-tuned) | ~82% |
| **Both combined (ensemble)** | **84%** |

Combining the two models gives a real accuracy boost over using either one by itself.

## How it works, in plain terms

- **The custom CNN** scans the image in layers, gradually picking up on patterns — edges, textures, colors — that matter for telling healthy coral (usually colorful) apart from bleached coral (usually pale/white). The attention mechanism (CBAM) helps it "zoom in" on the coral itself rather than getting distracted by background water or rocks.
- **VGG19** already knows how to recognize general visual patterns from prior training, so instead of starting from scratch, this project reuses that knowledge and just retrains the final layers to specialize in coral health.
- **The ensemble** simply averages both models' confidence scores and picks whichever class (healthy/bleached) scores higher overall — a classic way to make a more reliable prediction than trusting a single model.

## Dataset

- Binary classification: `Healthy` vs `Bleached`
- Images split into training, validation, and testing sets
- All images resized to 128×128 pixels and normalized before training
- Dataset: [Bleached Corals and Healthy Corals Classification (Kaggle)](https://www.kaggle.com/datasets)

## Project Structure

```
ReefHealth-AI/
├── notebooks/
│   └── DL_project.ipynb     # Full pipeline: load data → train → evaluate
├── src/
│   ├── models/
│   │   ├── cnn_cbam.py      # Custom CNN + attention model
│   │   ├── vgg19_model.py   # VGG19 transfer-learning model
│   │   └── ensemble.py      # Combines both models' predictions
│   ├── data_loader.py       # Loads and augments the images
│   └── evaluate.py          # Generates metrics, charts, ROC curves
├── results/                 # Saved charts and confusion matrices
├── requirements.txt
└── README.md
```

## Evaluation

The notebook generates:
- Precision, recall, and F1-score for each model
- ROC curves with AUC scores
- A confusion matrix for the ensemble
- A side-by-side comparison of prediction confidence across all three approaches

## Tech Stack

- **Framework:** TensorFlow / Keras
- **Models:** Custom CNN with CBAM attention, VGG19 (pre-trained on ImageNet)
- **Evaluation:** scikit-learn, matplotlib, seaborn
- **Data handling:** NumPy, Pandas, Keras ImageDataGenerator
