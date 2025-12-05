# MNIST Digit Classification

This project implements and compares two neural network models for the MNIST handwritten digit dataset:  
a baseline **Multilayer Perceptron (MLP)** and an improved **Convolutional Neural Network (CNN)**.  
The goal is to build a clean, reproducible pipeline for image classification while understanding how different
architectural choices and training strategies affect learning performance.

---

## 📌 Problem Statement

Handwritten digit recognition is a classic benchmark in machine learning.  
Given a 28×28 grayscale image representing a handwritten digit (0–9), the task is to build a model that
accurately predicts the correct class.

Although MNIST is simple, it captures large variations in handwriting style, thickness, curvature, and shape.
A successful classifier must therefore:

- extract meaningful patterns from raw pixel values,  
- generalise across diverse handwriting styles, and  
- train efficiently using modern deep-learning techniques.

This project focuses on:

1. Implementing two neural architectures: a fully connected MLP and a convolutional CNN.
2. Building a clean training and evaluation pipeline using PyTorch.
3. Understanding core DL concepts such as batching, activation functions, optimisers, and regularisation.
4. Visualising learning behaviour through loss/accuracy curves.
5. Producing a portfolio-ready implementation suitable for research or internship applications.

