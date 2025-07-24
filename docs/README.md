---
layout: default
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
  });
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS_HTML-full"></script>

# Adversarial Training and Randomized Smoothing

## Overview

This project explores two fundamental approaches to improving the robustness of neural networks against adversarial attacks: **Adversarial Training** and **Randomized Smoothing**. Both methods are implemented and evaluated on the MNIST dataset using a simple convolutional neural network (CNN) architecture.

- **Adversarial Training**: The model is trained not only on clean data but also on adversarially perturbed examples, aiming to minimize the worst-case loss within a specified perturbation set.
- **Randomized Smoothing**: A certified defense that constructs a new, smoothed classifier by averaging the predictions of the base classifier over random Gaussian perturbations of the input, providing probabilistic guarantees on robustness.

## Theoretical Background

### Adversarial Attacks

Given a classifier \( f: \mathbb{R}^d \to \{1, \ldots, K\} \), an adversarial attack seeks a small perturbation \( \delta \) such that \( f(x) \neq f(x + \delta) \), while \( \|\delta\| \leq \epsilon \) for some norm (e.g., \( \ell_2 \) or \( \ell_\infty \)). The **gradient attack** implemented here is a single-step projected gradient method:

\[
x_{\text{adv}} = \text{clip}(x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y)), 0, 1)
\]

where \( \mathcal{L} \) is the loss function (typically cross-entropy).

### Adversarial Training

Adversarial training solves the following robust optimization problem:

\[
\min_\theta \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \max_{\|\delta\| \leq \epsilon} \mathcal{L}(f_\theta(x + \delta), y) \right]
\]

This is approximated by generating adversarial examples on-the-fly during training and updating the model parameters to minimize the loss on these examples.

### Randomized Smoothing

Randomized smoothing constructs a new classifier \( g \) from a base classifier \( f \) as follows:

\[
g(x) = \arg\max_{c} \mathbb{P}(f(x + \eta) = c)
\]

where \( \eta \sim \mathcal{N}(0, \sigma^2 I) \). The smoothed classifier can **certify** robustness within an \( \ell_2 \) ball of radius

\[
r = \sigma \cdot \Phi^{-1}(p_A)
\]

where \( p_A \) is a lower confidence bound on the probability that class \( A \) is predicted, and \( \Phi^{-1} \) is the inverse standard normal CDF.

## Project Structure

- **main_training.py**: Standard training of the CNN on MNIST, evaluation on clean and adversarially perturbed data, and visualization of results.
- **main_adv_training.py**: Adversarial training using the gradient attack, saving the robust model, and evaluating its performance.
- **main_random_smoothing.py**: Trains a smoothed classifier using Gaussian noise, saving the base classifier for certification.
- **main_compare.py**: Loads all trained models (standard, adversarial, smoothed) and compares their certified robustness using randomized smoothing.
- **src/models.py**: Defines the CNN architecture (`ConvNN`) and the `SmoothClassifier` for randomized smoothing, including certification and prediction logic.
- **src/attacks.py**: Implements the gradient-based adversarial attack and a wrapper for generating adversarial examples.
- **src/training_and_evaluation.py**: Contains training loops, loss functions (including adversarial loss), prediction, and robustness evaluation routines.
- **src/utils.py**: Utility functions for loading MNIST data and selecting the computation device.

## Architecture

- **Model**: A simple convolutional neural network with two convolutional layers, batch normalization, and a fully connected output layer.
- **Training**: Supports both standard and adversarial training regimes.
- **Evaluation**: Includes clean accuracy, adversarial accuracy (under \( \ell_2 \) and \( \ell_\infty \) attacks), and certified robustness via randomized smoothing.

## Mathematical Guarantees

- **Adversarial Training**: Empirically increases robustness to attacks within the chosen norm and \( \epsilon \).
- **Randomized Smoothing**: Provides a certified radius \( r \) such that, with high probability, the classifier’s prediction is constant for all \( x' \) with \( \|x' - x\|_2 < r \).

## Usage

1. **Standard Training**: Run `main_training.py` to train a baseline model.
2. **Adversarial Training**: Run `main_adv_training.py` to train a robust model.
3. **Randomized Smoothing**: Run `main_random_smoothing.py` to train and save a smoothed classifier.
4. **Comparison**: Run `main_compare.py` to compare the certified robustness of all models.

## References

- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.
- Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing", ICML 2019. 