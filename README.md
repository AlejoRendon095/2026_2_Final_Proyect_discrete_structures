# Sentiment Analysis on GoEmotions (Reddit Comments)

## Project Overview

This project performs binary sentiment classification (positive and negative) on the GoEmotions dataset, which consists of Reddit comments annotated with fine-grained emotions. The objective is to explore and compare multiple deep learning architectures for sentiment analysis as part of **Integrative Task 2 – Discrete Mathematics and Structures III (2026-2)**.

## Objectives

- Preprocess and prepare the GoEmotions dataset for binary sentiment classification
- Implement and evaluate baseline and deep learning models
- Compare model performance using appropriate metrics
- Document findings and model behaviors

## Team Members

- [Team Member 1]
- [Team Member 2]
- [Team Member 3]

## Dataset

The **GoEmotions** dataset contains Reddit comments annotated with 27 emotion categories and a neutral class. For this project, we map these fine-grained emotions to binary sentiment labels (positive / negative) and train classifiers on the comment text.

## Models to Explore

The following architectures are planned for implementation and comparison:

- **DummyClassifier** (baseline)
- **Dense Neural Network** (feedforward)
- **Vanilla Recurrent Neural Network (RNN)**
- **Long Short-Term Memory (LSTM)**
- **Transformer-based model** (optional extension)

*Note: These models represent the planned scope of the project. Implementation and evaluation will be completed incrementally.*

## Repository Structure

```
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── saved_models/
├── docs/
├── prompts/
└── logs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Execution instructions will be added as the implementation progresses. Planned entry points include:

- Data preprocessing scripts
- Model training scripts
- Evaluation and visualization notebooks