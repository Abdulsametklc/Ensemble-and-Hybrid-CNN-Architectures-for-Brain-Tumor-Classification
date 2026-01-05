# 🧠 Ensemble and Hybrid CNN Architectures for Brain Tumor Classification

## 📌 Project Overview
This project presents a multi-model hybrid deep learning framework for brain tumor classification using MRI images.
Instead of relying on a single convolutional neural network (CNN), the study explores hybrid and ensemble-based
architectures by combining the strengths of multiple pretrained CNN backbones.

The main motivation is to improve classification accuracy, robustness, and generalization performance in medical
image analysis, where reliable diagnosis is critical.

---

## 🎯 Objectives
- Design hybrid CNN architectures using multiple pretrained models
- Improve performance compared to single-model approaches
- Perform comprehensive evaluation using medical-relevant metrics
- Provide a reproducible and extensible deep learning pipeline
- Analyze the impact of feature fusion in brain tumor MRI classification

---

## 🧠 Hybrid Model Concept

### Why Hybrid Models?
Single CNN architectures may:
- Focus on limited feature representations
- Miss complementary spatial or semantic features
- Overfit on medical datasets

Hybrid models aim to:
- Capture multi-scale and multi-level features
- Combine global and local representations
- Improve robustness and generalization

---

### Hybrid Architecture Strategy
Each hybrid model follows a common strategy:
1. Multiple pretrained CNN backbones are used in parallel
2. Feature maps are extracted independently
3. Extracted features are fused (concatenation or merging)
4. Fully connected layers perform final classification

Pretrained models are primarily used as feature extractors,
with optional fine-tuning depending on the experiment.

---

### Example Hybrid Model Combinations
- EfficientNetB0 + ResNet50
- DenseNet121 + EfficientNetB0
- VGG16 + VGG19
- EfficientNetB3 + ResNet101
- Custom baseline model (KilicNN)

Each model is trained and evaluated independently
to ensure fair and consistent comparison.

---

## 🏗️ Model Architecture Overview

### General Processing Flow
Input MRI Image
↓
Preprocessing & Normalization
↓
CNN Backbone A ─┐
├─ Feature Fusion
CNN Backbone B ─┘
↓
Fully Connected Layers
↓
Softmax Output


### Key Architectural Components
- Pretrained CNN backbones
- Adaptive Average Pooling
- Feature concatenation
- Dropout for regularization
- Softmax output layer

---

## 🧪 Dataset
- Modality: Brain MRI
- Task: Binary classification (Tumor / No Tumor)
- Preprocessing steps:
  - Image resizing
  - Normalization
  - Data augmentation (rotation, flipping, etc.)

Due to GitHub file size limitations, large datasets
and trained model weights are hosted externally.

---

## 📊 Evaluation Metrics
To ensure reliable medical evaluation, the following metrics are used:
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-score
- ROC-AUC
- Cohen’s Kappa
- Confusion Matrix

These metrics provide both statistical
and clinical relevance.

---

## 🛠️ Technologies & Tools

### Programming and Frameworks
- Python
- PyTorch
- Torchvision
- NumPy
- Pandas

### Training and Evaluation
- Scikit-learn (metrics)
- Matplotlib (visualization)
- CUDA (GPU acceleration)

### Experiment Management
- Google Colab
- Git & GitHub
- Git LFS (Large File Storage)

---

## 📁 Project Structure
├── data/
│ └── dataset (external)
├── models/
│ ├── hybrid_models/
│ └── baseline_models/
├── training/
│ └── train.py
├── evaluation/
│ └── metrics.py
├── results/
│ ├── plots/
│ └── confusion_matrices/
├── README.md
└── requirements.txt

---

## 🚀 Key Contributions
- Implementation of multiple hybrid CNN architectures
- Extensive comparative performance analysis
- Modular and extensible training pipeline
- High-performance results suitable for academic research

---

## 🔮 Future Work
- Multi-class brain tumor classification
- Explainable AI methods (Grad-CAM, SHAP)
- Transformer-based architectures
- Cross-dataset generalization
- Clinical decision support systems

---

## 👨‍💻 Author
Abdulsamet Kılıç  
Computer Engineering  
Artificial Intelligence and Deep Learning Researcher

---

## 📄 License
This project is intended for research and educational purposes only.

