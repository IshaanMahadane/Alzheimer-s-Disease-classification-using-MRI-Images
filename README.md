# 🧠 Alzheimer’s Disease Classification from MRI Images

A deep learning–based project that classifies MRI brain scans into different stages of Alzheimer’s Disease using **VGG16 Transfer Learning** and **TensorFlow**.  
The model achieves high accuracy and demonstrates the use of **transfer learning**, **class imbalance handling (SMOTE)**, and **visual interpretability** in medical imaging.

---

## 📘 Project Overview

Early detection of Alzheimer’s Disease is crucial for timely intervention and patient care.  
This project uses **Convolutional Neural Networks (CNNs)** and **transfer learning (VGG16)** to classify MRI scans into four categories:

- 🧩 Non-Demented  
- ⚠️ Very Mild Demented  
- ⚠️ Mild Demented  
- 🚨 Moderate Demented

The model was trained and evaluated on a dataset of **2,500+ MRI images**, achieving **~94% accuracy** on the test set.

---

## 🚀 Key Features

- **Transfer Learning (VGG16):** Pretrained model fine-tuned for MRI classification.
- **Data Preprocessing:** Resizing, normalization, augmentation, and encoding.
- **Class Imbalance Handling:** Used **SMOTE** to generate synthetic samples for minority classes.
- **Regularization:** Early stopping, dropout, and learning-rate scheduling.
- **Visualization:** Confusion matrix, loss and accuracy curves plotted with Matplotlib and Seaborn.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.

---

## 🧩 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **Frameworks** | TensorFlow, Keras |
| **Model Architecture** | VGG16 Transfer Learning |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Other Tools** | scikit-learn, imbalanced-learn (SMOTE) |

---

## 📊 Model Performance

| Metric | Score |
|--------|--------|
| **Training Accuracy** | ~95% |
| **Validation Accuracy** | ~93% |
| **Test Accuracy** | ~94% |
| **F1-Score (Weighted)** | 0.92 |

**Confusion Matrix:**  
- 1132 correctly classified Non-Demented cases  
- 771 correctly classified Very Mild Demented cases

---

## 🧠 Model Architecture

Input (224x224x3)
│
├── VGG16 base (frozen layers)
│
├── Flatten
├── Dense (256, ReLU)
├── Dropout (0.3)
├── Dense (128, ReLU)
├── Dropout (0.2)
└── Dense (4, Softmax)

---

## ⚙️ How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/alzheimers-mri-classification.git
   cd alzheimers-mri-classification

2. **Install dependencies**

   pip install -r requirements.txt   

3. **Place MRI images under dataset/ with subfolders:**

dataset/
  ├── train/
  │   ├── MildDemented/
  │   ├── ModerateDemented/
  │   ├── VeryMildDemented/
  │   └── NonDemented/
  └── test/

4. **Train the model**

python train.py

5. **Evaluate the model**

python evaluate.py
  
