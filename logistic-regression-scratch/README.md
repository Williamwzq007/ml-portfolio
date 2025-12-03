# Logistic Regression from Scratch

This project implements binary logistic regression **from scratch** using NumPy only.

The aim is to go from the **mathematical formulation** to a **working classifier**,  
including:

- Deriving the logistic regression model and the binary cross-entropy loss
- Computing the gradient analytically
- Implementing gradient descent optimisation
- Training and evaluating the model on a 2D synthetic dataset
- Visualising the loss curve and the learned decision boundary

---

## Project structure

- `logistic_regression_scratch.ipynb` – main notebook with all code, maths, plots and discussion
- `figures/` – saved figures (loss curve, decision boundary)
- `data/` – placeholder for future datasets (not used in the basic version)

---

## Main ideas

1. Represent data as a matrix \(X \in \mathbb{R}^{n_{\text{features}} \times m}\) and labels as \(y \in \{0,1\}^m\).
2. Use the logistic model
   \[
   \hat{y} = \sigma(w^\top x + b)
   \]
   with sigmoid activation \(\sigma(z) = \frac{1}{1 + e^{-z}}\).
3. Minimise the binary cross-entropy loss using **gradient descent**.
4. Evaluate the learned classifier in terms of accuracy and visualise the decision boundary in 2D.

---

## Possible extensions

- Compare different learning rates and initialisations
- Add L2 regularisation
- Compare with `sklearn.linear_model.LogisticRegression`
- Experiment on a real dataset (e.g. a binary subset of the Iris dataset)
