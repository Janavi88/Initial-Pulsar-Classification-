# Pulsar Signal Classification using Machine Learning

## Project Overview

This project uses machine learning to classify radio signal candidates as either
pulsars or non-pulsars using the HTRU2 dataset.

The project compares three supervised machine learning models:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

SMOTE is also used with Random Forest to handle the class imbalance in the training data.

## Dataset

The HTRU2 dataset contains 17,898 samples and 8 numerical features.

Target classes:

- 0 - Non-Pulsar
- 1 - Pulsar

The dataset contains 16,259 non-pulsar samples and 1,639 pulsar samples.

## Methods

1. Load and analyze the dataset
2. Split the data into training and testing sets
3. Scale the features using StandardScaler
4. Train Logistic Regression
5. Train SVM
6. Train Random Forest
7. Apply SMOTE to the training data
8. Train Random Forest using the SMOTE-balanced data
9. Compare model performance using accuracy, precision, recall and F1-score
10. Generate visualizations including class distribution and ROC curves

## How to Run

Create and activate a Python virtual environment:

```bash
python -m venv venv

Activate it on Windows:
venv\Scripts\activate

Install the required packages:
pip install -r requirements.txt

Run the project:
python data_analysis.py

Main Results
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
Recall is particularly important because correctly identifying actual pulsars is an important objective of the classification task.
Project Files
Pulsar_ML_project/
│
├── data/
│   └── HTRU_2.csv
│
├── data_analysis.py
├── requirements.txt
└── README.md


