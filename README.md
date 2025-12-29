# ml-portfolio

## ML Portfolio

Hi! I'm Ziqian Wang, a Mathematics undergraduate at Imperial College London.

This repository collects my machine learning projects and small experiments.  
A recurring theme is to **connect implementation with the underlying mathematics**  
(linear algebra, optimisation, and probability).

---

## Current projects

### 1. Logistic regression from scratch (NumPy)

A minimal binary classifier implemented **from first principles**:

- derive the logistic loss and its gradient;
- implement gradient descent (and variants) without high-level ML libraries;
- visualise the decision boundary and the effect of learning rate / iterations.

This project is mainly about understanding logistic regression as

\[
\text{a parametric probabilistic model trained by gradient-based optimisation}.
\]

---

### 2. MNIST classification with PyTorch

Handwritten digit classification on MNIST using both an MLP and a small CNN:

- full training pipeline in PyTorch: dataloaders, normalisation, training/validation loops;
- compare an MLP on flattened images with a convolutional network that uses spatial structure;
- use cross-entropy as negative log-likelihood (MLE) and visualise learning curves.

The notebook includes **lecture-style notes** linking the code to:

- affine maps and ReLU as piecewise-linear transformations;
- softmax + cross-entropy as a probabilistic model;
- basic optimisation considerations (normalisation, conditioning, choice of optimiser).

---

## Next steps

More projects will be added over time, including:

- small optimisation experiments (e.g. GD vs SGD vs Adam);
- further models that combine mathematical structure with practical ML.
