# Naive Bayes Sentiment Classifier From Scratch

A simple implementation of a Multinomial Naive Bayes sentiment classifier built from scratch in Python.

## Overview

This project implements a complete Naive Bayes text classification pipeline without using machine learning libraries such as scikit-learn.

The goal of the project is to understand the mathematical and algorithmic foundations behind probabilistic NLP models.

## Features

* Text preprocessing
* Vocabulary construction
* Word-frequency counting
* Prior probability calculation
* Laplace smoothing
* Log-probability scoring
* Sentiment prediction
* Model evaluation

## Concepts Covered

* Bayes Theorem
* Prior, Likelihood, and Posterior Probabilities
* Conditional Probability
* Multinomial Naive Bayes
* Laplace Smoothing
* Log Probabilities
* Text Classification

## Project Pipeline

```text
Training Data
    ↓
Preprocessing
    ↓
Word Frequency Counting
    ↓
Vocabulary Construction
    ↓
Prior Calculation
    ↓
Laplace Smoothing
    ↓
Sentence Scoring (Log Probabilities)
    ↓
Prediction
    ↓
Evaluation
```

## Example

Input:

```text
I love this movie
```

Output:

```text
positive
```

## Technologies

* Python
* Jupyter Notebook

## Learning Objective

This project was created as a learning exercise to gain a deeper understanding of probabilistic Natural Language Processing and the inner workings of Naive Bayes classifiers.
