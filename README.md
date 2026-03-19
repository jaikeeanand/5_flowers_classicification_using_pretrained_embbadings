# 5_flowers_classicification_using_pretrained_embbadings
Developed a 5-class flower image classifier using MobileNetV2 with transfer learning. Built a tf.data pipeline for image loading and preprocessing from CSV. Trained using Adam and tracked experiments with W&amp;B, achieving efficient training and improved accuracy.
# 5 Flower Classification using Transfer Learning

## Overview

This project focuses on multi-class image classification of five flower types: Lilly, Tulip, Sunflower, Lotus, and Orchid. A transfer learning approach is used with MobileNetV2 pretrained on ImageNet to efficiently extract features and improve model performance.

---

## Objectives

* Classify flower images into 5 categories
* Use pretrained embeddings for better accuracy
* Build an efficient data pipeline using TensorFlow
* Track experiments using Weights & Biases (W&B)

---

## Methodology

### Model

* Base Model: MobileNetV2 (pretrained on ImageNet)
* Frozen convolutional layers (feature extraction)
* Custom classification head (Dense layer with softmax)

### Data Pipeline

* Image paths and labels stored in CSV files
* Loaded using `tf.data`
* Steps:

  * Read image
  * Decode
  * Resize (e.g., 128×128)
  * Normalize
  * Batch & Prefetch

### Training

* Loss: Sparse Categorical Crossentropy
* Optimizer: Adam
* Metrics: Accuracy

---

## Experiment Tracking

* Integrated **Weights & Biases (W&B)**
* https://wandb.ai/jaicky-iit-ism-dhanbad/5_Flowers_Classification_Transfer_Learning1/runs/so4xlabj/panel/m1rwz34ns?
* https://wandb.ai/jaicky-iit-ism-dhanbad/5_Flowers_Classification_Transfer_Learning1/runs/so4xlabj/panel/6ke39bk7s?
  
* Features:

  * Loss & accuracy visualization
  * Run comparison

---

## Dataset Structure

```
flower_images/
│
├── Lilly/
├── Tulip/
├── Sunflower/
├── Lotus/
└── Orchid/
```

CSV format:

```
filepath,label
path/to/image.jpg,Lilly
```

---

##  How to Run

### 1️ Install dependencies

```
pip install tensorflow wandb pandas scikit-learn
```

### 2️⃣ Login to W&B

```
wandb login
```

### 3️⃣ Run training

```
python [5_flower_classification_using_pretrained_embedding](https://colab.research.google.com/drive/1YbboNTwp6wKYPwxuoJhKiXvR8XM1SFeD#scrollTo=QU24nOJq9xPj).py
```

---

## Results

* Efficient training using transfer learning
* Improved accuracy compared to training from scratch
* Faster convergence

---

## Key Concepts

* Transfer Learning
* Pretrained Embeddings
* TensorFlow tf.data pipeline
* Experiment Tracking (W&B)

---

## Future Work

* Fine-tuning MobileNet layers
* Data augmentation
* Trying other architectures (ResNet, EfficientNet)

---

## 👨‍🎓 Author

Avinash Kumar Bharti
