# Logistic Regression from Scratch

This project implements binary logistic regression **from scratch** using NumPy only.  
The goal is to go from the **mathematical formulation** of the model to a **working classifier**,  
including optimisation with gradient descent and visualisation of the learned decision boundary.

---

## 📘 Overview

In this project, I:

- Derived the logistic regression model and the binary cross-entropy loss  
- Computed gradients $\partial J / \partial w$ and $\partial J / \partial b$ analytically  
- Implemented gradient descent manually  
- Generated a 2D synthetic dataset  
- Visualised the loss curve and the final decision boundary  
- Evaluated the model accuracy  

This project demonstrates the connection between **mathematical modelling** and **practical machine learning implementation**, which is especially relevant for applied mathematics & ML research.

---

## 🧠 Mathematical Formulation

Given:

- Input vectors $x^{(i)} \in \mathbb{R}^n$, for $i = 1, \dots, m$
- Binary labels $y^{(i)} \in \{0, 1\}$

We stack them as:

- Data matrix $X \in \mathbb{R}^{n \times m}$  
- Label row vector $Y \in \{0, 1\}^{1 \times m}$  

### Logistic model

The prediction is

$$
\hat{y}^{(i)} = \sigma(w^\top x^{(i)} + b),
$$

where the sigmoid function is defined as

$$
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

### Loss function

The binary cross-entropy loss is

$$
J(w, b) = -\frac{1}{m} 
\sum_{i=1}^m 
\Big[
y^{(i)} \log(\hat{y}^{(i)}) +
(1 - y^{(i)}) \log(1 - \hat{y}^{(i)})
\Big].
$$

### Gradient

Let $A = \hat{Y} = \sigma(w^\top X + b)$.

Then:

$$
\frac{\partial J}{\partial w}
= \frac{1}{m} X (A - Y)^\top,
\qquad
\frac{\partial J}{\partial b}
= \frac{1}{m} \sum_{i=1}^m \big( \hat{y}^{(i)} - y^{(i)} \big).
$$

### Gradient descent update

$$
w \leftarrow w - \alpha \frac{\partial J}{\partial w},
\qquad
b \leftarrow b - \alpha \frac{\partial J}{\partial b},
$$

where $\alpha$ is the learning rate.

---

## 📂 Project Structure

