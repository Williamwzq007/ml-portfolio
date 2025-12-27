# MNIST Digit Classification

This project implements and compares two neural network models for handwritten digit recognition on the MNIST dataset:  
a baseline **Multilayer Perceptron (MLP)** and an improved **Convolutional Neural Network (CNN)**.

The goal is to build a clean and reproducible image-classification pipeline, while analysing how different architectural choices and training strategies affect learning performance.

---

## 📌 Problem Statement

Handwritten digit recognition is a classic benchmark in machine learning.  
Given a 28×28 grayscale image representing a handwritten digit (0–9), the task is to build a model that accurately predicts the correct class.

Although MNIST is considered a relatively simple dataset, it exhibits substantial variation in handwriting style, stroke thickness, curvature, and overall shape.  
A successful classifier must therefore:

- extract meaningful patterns from raw pixel values,
- generalise across diverse handwriting styles, and
- train efficiently using modern deep-learning techniques.

This project focuses on:

1. Implementing two neural architectures: a fully connected MLP as a baseline and a convolutional CNN as an improved model.
2. Building a clean and modular training and evaluation pipeline using PyTorch.
3. Understanding core deep-learning concepts such as batching, activation functions, optimisers, and regularisation.
4. Analysing learning behaviour through training and test loss/accuracy curves.
5. Producing a portfolio-ready implementation suitable for research or internship applications.

---

## 🧠 Methods Overview

- **Data**: MNIST handwritten digit dataset (60,000 training samples, 10,000 test samples)  
- **Preprocessing**: Conversion to tensors and input normalisation  
- **Baseline model**: Multilayer Perceptron (MLP) operating on flattened image inputs  
- **Improved model**: Convolutional Neural Network (CNN) exploiting local spatial structure  
- **Optimisation**: Adam optimiser with cross-entropy loss  
- **Evaluation**: Test accuracy and loss tracked across epochs  

---

## 📈 Results

The MLP provides a strong but limited baseline, achieving high accuracy while ignoring spatial relationships between pixels.  
The CNN significantly outperforms the MLP, converging faster and achieving higher test accuracy by learning hierarchical, local features through convolution and pooling.

Training and evaluation curves clearly demonstrate the performance gap between the two architectures.

---

## ✅ Conclusion

In this project, we implemented and evaluated two neural network architectures for handwritten digit recognition on the MNIST dataset: a multilayer perceptron (MLP) baseline and an improved convolutional neural network (CNN).

The MLP demonstrated how fully connected networks can learn from flattened pixel representations, but its performance was limited by the absence of spatial awareness.  
By contrast, the CNN effectively exploited the spatial structure of images, learning local and hierarchical features such as edges and strokes, which led to faster convergence and improved generalisation.

These results highlight the importance of architectural design in image-classification tasks. While MLPs provide useful baselines, convolutional networks are far better suited for visual data due to their parameter efficiency and ability to capture spatial patterns.

Overall, this project demonstrates a complete and reproducible deep learning workflow—from data preprocessing and model design to training, evaluation, and visualisation—and provides a solid foundation for extending to more complex datasets and architectures.

---

## 🔮 Future Work

Possible extensions include experimenting with batch normalisation, alternative optimisers, and deeper convolutional architectures, as well as applying the pipeline to more challenging datasets such as Fashion-MNIST or CIFAR-10.



