# gradient-descent-and-logistic-regression

## Overview

This project implements gradient descent optimization and logistic regression with regularization, focusing on modular design, extensibility, and performance analysis. It was developed as part of a university coursework and includes:

- Gradient descent with customizable learning rate strategies.
- Regularized logistic regression supporting L1 and L2 penalties.
- Visualizations of optimization behavior and model performance.

## Features

1. **Gradient Descent**
   - **Learning Rate Strategies**: Supports fixed and exponentially decaying learning rates.
   - **Objective Functions**: Includes implementations for L1 and L2 regularization.
   - **Efficiency**: Optimized to avoid storing all intermediate solutions, with support for callback functions to track progress.

2. **Logistic Regression**
   - **Regularization**: Implements L1 and L2 penalties via a modular design.
   - **Hyperparameter Tuning**: Cross-validation for selecting optimal regularization strength.
   - **Performance Metrics**: Generates ROC curves and evaluates model accuracy.

3. **Investigative Tools**
   - Analyze and visualize the behavior of gradient descent on different objectives.
   - Compare convergence rates for varying learning rates.

## File Structure

- **`modules.py`**: Core modules for L1, L2, logistic regression, and regularized objectives.
- **`gradient_descent.py`**: Implements the gradient descent algorithm.
- **`learning_rate.py`**: Contains learning rate strategies (`FixedLR`, `ExponentialLR`).
- **`logistic_regression.py`**: Logistic regression implementation with support for penalties.
- **`gradient_descent_investigation.py`**: Scripts for analyzing gradient descent behavior.
