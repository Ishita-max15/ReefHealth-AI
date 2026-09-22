# Coral Reef Health Classification 🪫

> Ensemble deep learning model combining a Custom CNN with CBAM attention and VGG19 transfer learning for binary classification of **bleached vs healthy coral reefs**.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)


---

## 📊 Results

| Model | Test Accuracy |
|-------|---------------|
| Custom CNN + CBAM | ~80% |
| VGG19 (Transfer Learning) | ~82% |
| **Ensemble (Avg Voting)** | **84%** |

---

## 🧠 Architecture Overview

### 1. Custom CNN with CBAM Attention
- Conv2D layers (32 → 64 filters) with ReLU activation
- **CBAM (Convolutional Block Attention Module)** after each conv block:
  - **Channel Attention**: GlobalAvgPool + GlobalMaxPool → shared Dense → sigmoid
    - **Spatial Attention**: mean + max along channel axis → Conv2D → sigmoid
    - Dense(128) → sigmoid output

    ### 2. VGG19 (Transfer Learning)
    - Pre-trained on ImageNet, top layers frozen
    - Custom head: Flatten → Dense(256) → sigmoid output
    - Fine-tuned with Adam optimizer (lr=0.001)

    ### 3. Ensemble
    - Average probability voting: `(CNN_prob + VGG19_prob) / 2`
    - Final prediction: threshold at 0.5

    ---

    ## 📁 Dataset

    **Bleached Corals and Healthy Corals Classification**
    - Binary classification: `Healthy` vs `Bleached`
    - Split into `Training/`, `Validation/`, `Testing/` directories
    - Input image size: 128x128 pixels
    - Normalized: pixel values rescaled to [0, 1]

    > Dataset available on Kaggle: [Bleached Corals and Healthy Corals](https://www.kaggle.com/datasets)

    ---

    ## 📂 Project Structure

    ```
    ReefHealth-AI/
    ├── notebooks/
    │   └── DL_project.ipynb          # Main notebook with full pipeline
    ├── src/
    │   ├── models/
    │   │   ├── cnn_cbam.py               # Custom CNN + CBAM architecture
    │   │   ├── vgg19_model.py            # VGG19 transfer learning model
    │   │   └── ensemble.py               # Ensemble voting logic
    │   ├── data_loader.py             # Data loading & augmentation
    │   └── evaluate.py                # Metrics, plots, ROC curves
    ├── results/                       # Confusion matrix, ROC curve images
    ├── requirements.txt
    ├── .gitignore
    └── README.md
    ```

    ---

    ## 📈 Evaluation & Visualizations

    The notebook includes:
    - Per-class **Precision, Recall, F1-Score** bar charts for CNN and VGG19
    - **ROC Curves** with AUC scores for all three models
    - **Confusion Matrix** heatmap for the Ensemble model
    - **Prediction Probability Comparison** across CNN, VGG19, and Ensemble

    ---

    ## 🛠️ Tech Stack

    - **Framework**: TensorFlow / Keras
    - **Models**: Custom CNN, VGG19 (ImageNet weights)
    - **Attention**: CBAM (Channel + Spatial Attention)
    - **Evaluation**: scikit-learn, matplotlib, seaborn
    - **Data**: NumPy, Pandas, Keras ImageDataGenerator

    
