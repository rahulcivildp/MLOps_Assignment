# Boston Housing Price Prediction using Machine Learning

## Overview

This project implements an end-to-end machine learning workflow for predicting house prices using the Boston Housing dataset. The project demonstrates data loading, preprocessing, model training, evaluation, model persistence, and automated experimentation using multiple regression models.

## Dataset

The Boston Housing dataset was originally available through Scikit-learn but has since been deprecated. The dataset is loaded directly from the original source and contains housing-related features such as crime rate, average number of rooms, property tax rate, and other socio-economic indicators.

### Target Variable

* **MEDV**: Median value of owner-occupied homes (in $1000s)

### Features

* CRIM
* ZN
* INDUS
* CHAS
* NOX
* RM
* AGE
* DIS
* RAD
* TAX
* PTRATIO
* B
* LSTAT

---

## Project Structure

```text
Boston_Housing/
│
├── data_loader.py
├── misc.py
├── train.py
├── train2.py
├── requirements.txt
├── README.md
│
├── saved_models/
│
├── results/
│
└── .github/
    └── workflows/
        └── ml_pipeline.yml
```

---

## Models Implemented

### main Branch

* Readme.md

### dtree Branch

* Decision Tree Regressor

### kernelridge Branch

* Decision Tree Regressor
* Kernel Ridge Regressor

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd Boston_Housing
```

### Create a Virtual Environment (Optional)

Using Conda:

```bash
conda create -n digit python=3.11
conda activate digit
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train and Evaluate Decision Tree Model

```bash
python train.py
```

Example Output:

```text
Average MSE Score: 10.4161
```

---

### Train and Evaluate Kernel Ridge Model

```bash
python train2.py
```

Example Output:

```text
Average MSE Score: 20.2850
```

---

## Saved Models

Trained models are automatically stored inside:

```text
saved_models/
```

Example:

```text
saved_models/
├── Decision Tree Regressor.pkl
├── Kernel Ridge Regressor.pkl
└── scaler.pkl
```

---

## Evaluation Metrics

The following metrics are reported:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## GitHub Actions

A GitHub Actions workflow is configured for the `kernelridge` branch.

On every push to this branch, GitHub Actions will:

1. Check out the repository.
2. Install project dependencies.
3. Run `train.py`.
4. Run `train2.py`.
5. Display model performance in the workflow logs.

---

## Scribe

Tanmay Sarkar G25AI1050

Machine Learning Operations (MLOps) Assignment
