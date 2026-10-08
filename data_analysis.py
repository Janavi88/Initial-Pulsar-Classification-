import pandas as pd

# Load the HTRU2 dataset
df = pd.read_csv("data/HTRU_2.csv")

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
print(df["target"].value_counts())

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)

from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)

from sklearn.linear_model import LogisticRegression

# Create the Logistic Regression model
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

# Train the model
logistic_model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully!")

# Make predictions on the test data
y_pred = logistic_model.predict(X_test_scaled)

print("\nFirst 20 predictions:")
print(y_pred[:20])

print("\nFirst 20 actual values:")
print(y_test.iloc[:20].values)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nLogistic Regression Results:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

from sklearn.svm import SVC

# Create the SVM model
svm_model = SVC(
    kernel="rbf",
    random_state=42
)

# Train the SVM model
svm_model.fit(X_train_scaled, y_train)

print("\nSVM model trained successfully!")

# Make predictions using the SVM model
svm_pred = svm_model.predict(X_test_scaled)

print("\nFirst 20 SVM predictions:")
print(svm_pred[:20])

# Evaluate the SVM model
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)

print("\nSVM Results:")
print("Accuracy :", round(svm_accuracy, 4))
print("Precision:", round(svm_precision, 4))
print("Recall   :", round(svm_recall, 4))
print("F1 Score :", round(svm_f1, 4))

# Create SVM confusion matrix
svm_cm = confusion_matrix(y_test, svm_pred)

print("\nSVM Confusion Matrix:")
print(svm_cm)

# ============================================================
# RANDOM FOREST
# ============================================================

from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest model
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=110,
    min_samples_split=5,
    max_features=4,
    random_state=42,
    n_jobs=-1
)

# Train the Random Forest model
random_forest_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# Make predictions on the test data
rf_pred = random_forest_model.predict(X_test)

print("\nFirst 20 Random Forest predictions:")
print(rf_pred[:20])

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\nRandom Forest Results:")
print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1 Score :", round(rf_f1, 4))

# Create Random Forest confusion matrix
rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-" * 68)

print(f"{'Logistic Regression':<20} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
print(f"{'SVM':<20} {svm_accuracy:<12.4f} {svm_precision:<12.4f} {svm_recall:<12.4f} {svm_f1:<12.4f}")
print(f"{'Random Forest':<20} {rf_accuracy:<12.4f} {rf_precision:<12.4f} {rf_recall:<12.4f} {rf_f1:<12.4f}")

# ============================================================
# HANDLE CLASS IMBALANCE USING SMOTE
# ============================================================

from imblearn.over_sampling import SMOTE

print("\n" + "=" * 60)
print("CLASS IMBALANCE BEFORE SMOTE")
print("=" * 60)

print("Class 0:", sum(y_train == 0))
print("Class 1:", sum(y_train == 1))

# Apply SMOTE only to the training data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION AFTER SMOTE")
print("=" * 60)

print("Class 0:", sum(y_train_smote == 0))
print("Class 1:", sum(y_train_smote == 1))

print("\nTraining data shape after SMOTE:", X_train_smote.shape)

# ============================================================
# RANDOM FOREST WITH SMOTE
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST WITH SMOTE")
print("=" * 60)

# Create the Random Forest model
rf_smote_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train using the SMOTE-balanced training data
rf_smote_model.fit(X_train_smote, y_train_smote)

print("\nRandom Forest with SMOTE trained successfully!")

# Make predictions on the ORIGINAL test set
rf_smote_pred = rf_smote_model.predict(X_test_scaled)

# Calculate evaluation metrics
rf_smote_accuracy = accuracy_score(y_test, rf_smote_pred)
rf_smote_precision = precision_score(y_test, rf_smote_pred)
rf_smote_recall = recall_score(y_test, rf_smote_pred)
rf_smote_f1 = f1_score(y_test, rf_smote_pred)

print("\nRandom Forest with SMOTE Results:")
print("Accuracy  :", round(rf_smote_accuracy, 4))
print("Precision :", round(rf_smote_precision, 4))
print("Recall    :", round(rf_smote_recall, 4))
print("F1 Score  :", round(rf_smote_f1, 4))

# Confusion matrix
rf_smote_cm = confusion_matrix(y_test, rf_smote_pred)

print("\nRandom Forest with SMOTE Confusion Matrix:")
print(rf_smote_cm)

# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(
    f"{'Model':<25}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1 Score':<12}"
)

print("-" * 75)

print(
    f"{'Logistic Regression':<25}"
    f"{accuracy:<12.4f}"
    f"{precision:<12.4f}"
    f"{recall:<12.4f}"
    f"{f1:<12.4f}"
)

print(
    f"{'SVM':<25}"
    f"{svm_accuracy:<12.4f}"
    f"{svm_precision:<12.4f}"
    f"{svm_recall:<12.4f}"
    f"{svm_f1:<12.4f}"
)

print(
    f"{'Random Forest':<25}"
    f"{rf_accuracy:<12.4f}"
    f"{rf_precision:<12.4f}"
    f"{rf_recall:<12.4f}"
    f"{rf_f1:<12.4f}"
)

print(
    f"{'Random Forest + SMOTE':<25}"
    f"{rf_smote_accuracy:<12.4f}"
    f"{rf_smote_precision:<12.4f}"
    f"{rf_smote_recall:<12.4f}"
    f"{rf_smote_f1:<12.4f}"
)

print("=" * 75)


# ============================================================
# VISUALIZATION 1 - CLASS DISTRIBUTION
# ============================================================

import matplotlib.pyplot as plt

# Count the two classes
class_counts = df["target"].value_counts().sort_index()

# Create the bar chart
plt.figure(figsize=(7, 5))

plt.bar(
    ["Non-Pulsar (0)", "Pulsar (1)"],
    class_counts.values
)

plt.title("Pulsar Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")

# Display the values above each bar
for i, value in enumerate(class_counts.values):
    plt.text(i, value + 300, str(value), ha="center")

plt.tight_layout()

# Save the graph
plt.savefig("class_distribution.png", dpi=300)

# Display the graph
plt.show()

# ============================================================
# VISUALIZATION 2 - PROFILE MEAN DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df[df["target"] == 0]["profile_mean"],
    bins=50,
    alpha=0.6,
    label="Non-Pulsar"
)

plt.hist(
    df[df["target"] == 1]["profile_mean"],
    bins=50,
    alpha=0.6,
    label="Pulsar"
)

plt.title("Distribution of Profile Mean")
plt.xlabel("Profile Mean")
plt.ylabel("Number of Samples")
plt.legend()

plt.tight_layout()

# Save the graph
plt.savefig("profile_mean_distribution.png", dpi=300)

plt.show()

# ============================================================
# VISUALIZATION 3 - ROC CURVES
# ============================================================

from sklearn.metrics import roc_curve, roc_auc_score

# Get probability scores from Logistic Regression
lr_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]

# Get probability scores from Random Forest
rf_prob = random_forest_model.predict_proba(X_test_scaled)[:, 1]

# Get probability scores from Random Forest + SMOTE
rf_smote_prob = rf_smote_model.predict_proba(X_test_scaled)[:, 1]

# SVM needs probability=True to generate probability scores.
# So we will train a separate SVM specifically for the ROC curve.
svm_roc_model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

svm_roc_model.fit(X_train_scaled, y_train)

svm_prob = svm_roc_model.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
rf_smote_fpr, rf_smote_tpr, _ = roc_curve(y_test, rf_smote_prob)

# Calculate AUC scores
lr_auc = roc_auc_score(y_test, lr_prob)
svm_auc = roc_auc_score(y_test, svm_prob)
rf_auc = roc_auc_score(y_test, rf_prob)
rf_smote_auc = roc_auc_score(y_test, rf_smote_prob)

# Plot ROC curves
plt.figure(figsize=(8, 6))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_auc:.4f})"
)

plt.plot(
    svm_fpr,
    svm_tpr,
    label=f"SVM (AUC = {svm_auc:.4f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.4f})"
)

plt.plot(
    rf_smote_fpr,
    rf_smote_tpr,
    label=f"Random Forest + SMOTE (AUC = {rf_smote_auc:.4f})"
)

# Random guessing line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guess"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

# Save the graph
plt.savefig("roc_curve_comparison.png", dpi=300)

plt.show()