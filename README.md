# CNN Image Classification and Comparison with State-of-the-Art Models
This project was completed as part of the EN3150 Assignment 03.

This repository contains the implementation of a simple Convolutional Neural Network (CNN) for image classification and a comparative analysis with state-of-the-art pre-trained models: DenseNet and VGG.

## Project Overview

The project focuses on developing a simple CNN for image classification and enhancing the results by leveraging transfer learning with pre-trained models. The key objectives include:

- Building and training a custom CNN architecture.
- Fine-tuning state-of-the-art pre-trained models (DenseNet and VGG) for the same dataset.
- Evaluating and comparing the performance of these models.

---

## Dataset

The dataset used was Realwaste data set from the UCI Machine Learning Repository. It was split into training, validation, and testing subsets in the ratio of 60:20:20.

---

## Custom CNN Architecture

A custom CNN was implemented with the following design:
1. Convolutional layers with activation functions.
2. MaxPooling layers for dimensionality reduction.
3. Dropout layers to reduce overfitting.
4. Fully connected layers for classification.

**Training Details:**
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 20
- Learning Rates Tested: 0.001

---

## Transfer Learning with State-of-the-Art Models

Two state-of-the-art models, DenseNet and VGG, were fine-tuned for this task. These models were pre-trained on the ImageNet dataset and adapted to the classification task.

**Training Details:**
- Fine-tuned using the same dataset splits.
- Comparison based on test accuracy and training/validation losses.

---

## Results

The following metrics were evaluated for each model:
- Accuracy
- Confusion Matrix
- Precision and Recall

Then plots for training and validation loss were generated to compare learning rates and training stability.

---

## Discussion

### Trade-offs and Observations:
- **Custom CNN:** Simpler architecture, lightweight, but required significant training time to achieve reasonable accuracy.
- **DenseNet and VGG:** Pre-trained models provided superior accuracy and generalization but required higher computational resources.

---

## Conclusion

DenseNet and VGG demonstrated significant performance advantages due to their pre-trained weights and deeper architectures. However, the custom CNN provided valuable insights into model design and training.

---

