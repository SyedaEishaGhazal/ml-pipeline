# Task 2: Automate an End-to-End Machine Learning Pipeline Using GitHub Actions

## Project Overview

This repository implements and automates a complete machine learning workflow using GitHub Actions. The workflow includes data preprocessing, model training, testing, and saving the trained model as an artifact.

---

## Workflow Details

### 1. Machine Learning Pipeline - Github

- **Dataset:** California Housing dataset (public dataset).
- **Preprocessing:** Handling missing values, normalization/scaling of features.
- **Model Training:** Train a regression model using scikit-learn (e.g., RandomForestRegressor or any preferred model).
- **Model Saving:** Save the trained model using `joblib`.

### 2. Testing

- At least **2 unit tests** for data preprocessing functions.
- At least **1 test** for the ML model’s performance (e.g., check if R² score > 0.8).

### 3. GitHub Actions Workflow (`.github/workflows/ml_pipeline.yml`)

- Runs on every **push** or **pull request**.
- Steps include:
  - Set up Python environment.
  - Install dependencies.
  - Run unit tests.
  - Train the ML model.
  - Save the trained model as a GitHub Actions artifact.

---

## How to Use This Repository

### Prerequisites

- Python 3.8+
- Git
- GitHub account

### Setup Instructions

1. **Clone the repository:**

