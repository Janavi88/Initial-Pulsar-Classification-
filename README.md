# Pulsar Signal Classification using Machine Learning

## Project Overview
This project uses machine learning to classify radio signal candidates as **pulsars** or **non-pulsars** using the HTRU2 dataset.

## Models Used
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Random Forest with SMOTE for handling class imbalance

## Dataset
- **Samples:** 17,898
- **Features:** 8 numerical features
- **Non-pulsars (0):** 16,259
- **Pulsars (1):** 1,639

## Methodology
1. Load and analyze the dataset.
2. Split data into training and testing sets.
3. Scale features using `StandardScaler`.
4. Train Logistic Regression, SVM, and Random Forest models.
5. Apply SMOTE to balance the training data and retrain Random Forest.
6. Compare models using accuracy, precision, recall, F1-score, and ROC-AUC.
7. Visualize class distribution and ROC curves.

## How to Run

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows, activate it:

```bash
venv\Scripts\activate
```

Install dependencies and run the project:

```bash
pip install -r requirements.txt
python data_analysis.py
```

## Main Results
Models are evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. **Recall is particularly important** because identifying genuine pulsar candidates is a key objective.

## Project Structure

```text
Pulsar_ML_project/
├── data/
│   └── HTRU_2.csv
├── data_analysis.py
├── requirements.txt
└── README.md
```
