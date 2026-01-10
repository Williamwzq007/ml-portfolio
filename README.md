# ml-portfolio

## ML Portfolio

Hi! I'm Ziqian (William) Wang, a Mathematics undergraduate at Imperial College London, interested in Deep Learning, Natural Language Processing and Large Language Models.


This repository, maintained actively and updated regularly, serves as a living record of my machine learning projects and recent progress.
I started from the basics and continue to learn by building and extending models through hands-on experimentation, with new projects added as my interests and understanding evolve.

---

## Current projects

### 1. Logistic regression from scratch (NumPy)

A minimal binary classifier implemented **from first principles**:

- derive the logistic loss and its gradient;
- implement gradient descent (and variants) without high-level ML libraries;
- visualise the decision boundary and the effect of learning rate / iterations.

This project is mainly about understanding logistic regression
as a parametric probabilistic model trained by gradient-based optimisation.

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
### 3. NLP disaster tweets classification (TF-IDF + Transformers)

Binary text classification on short social media messages related to disaster events:

- establish a classical baseline using TF-IDF features with logistic regression;
- fine-tune a transformer-based model (DistilBERT) using the Hugging Face Trainer API;
- compare traditional linear models with contextualised language representations.

The project focuses on building an end-to-end NLP pipeline, including text preprocessing,
model training, validation using F1 score, and reproducible inference for submission.

Rather than leaderboard optimisation, the emphasis is on understanding the modelling choices
and trade-offs between classical machine learning approaches and modern transformer architectures.


## Next steps

More projects will be added over time, including:

- small optimisation experiments (e.g. GD vs SGD vs Adam);
- further models that combine mathematical structure with practical ML.
