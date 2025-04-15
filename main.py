import os
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# Load datasets
df_causal = pd.read_excel('data/causal_info.xlsx')
df_dsw = pd.read_excel('data/dsw.xlsx')
print("Columns in df_causal:", df_causal.columns)
print("Columns in df_dsw:", df_dsw.columns)

# Merge datasets
df = pd.merge(df_causal, df_dsw, how="left", on=["mir", "disease", "pmid"])

# Print the columns after merging
print("Columns after merging:", df.columns)

# Replace target labels with numeric values
if 'causality' in df.columns:
    df['causality'] = df['causality'].replace({'no': 0, 'yes': 1})
else:
    print("Causality column not found!")

# Drop columns if they exist
if 'mesh_Name' in df.columns:
    df = df.drop(['mesh_Name'], axis=1)

# Check for and drop NaN values
print("Initial NaN counts:\n", df.isnull().sum())
df.dropna(inplace=True)
print("NaN counts after dropping:\n", df.isnull().sum())

# Limit dataset to specific diseases
diseases = [ 'Alopecia', 'Alzheimer Disease', 'Asthma', 'Brain Neoplasms', 
             'Cataract', 'Down Syndrome', 'Endometriosis', 'Gastric Neoplasms', 
             'Glaucoma', 'Heart Failure', 'Hypertension', 'Ischemic Stroke', 
             'Kidney Failure', 'Leukemia', 'Lung Neoplasms', 'Melanoma', 
             'Pancreatic Neoplasms', 'Parkinson Disease', 'Schizophrenia', 
             'Stroke' ]
df = df[df['disease'].isin(diseases)]

# Convert target labels to numeric
df['causality'] = df['causality'].replace({'no': 0, 'yes': 1})

# Split dataset into features and labels
labels = df['causality']
features = df.drop(['causality'], axis=1)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.05, random_state=42)

# Encode categorical variables
encoder = ce.OneHotEncoder(cols=['category_x', 'mir', 'disease'])  # Correct column names
features_train = encoder.fit_transform(features_train)
features_test = encoder.transform(features_test)

# Print the resulting columns after encoding to verify
print("Features Train Columns:", features_train.columns)
print("Features Test Columns:", features_test.columns)

# Drop non-numeric columns that cannot be scaled
non_numeric_columns = ['mesh_name', 'pmid', 'category_y', 'description']
features_train = features_train.drop(non_numeric_columns, axis=1)
features_test = features_test.drop(non_numeric_columns, axis=1)

# Scale the data
scaler = RobustScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Model 1: Gaussian Naive Bayes
clf_nb = GaussianNB()
clf_nb.fit(features_train, labels_train)
pred_nb = clf_nb.predict(features_test)
score_nb = accuracy_score(labels_test, pred_nb) * 100
print("Naive Bayes Accuracy: {:.3f}%".format(score_nb))

# Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(labels_test, pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=[0, 1])
disp_nb.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Naive Bayes')
plt.show()

# Model 2: Support Vector Machine
clf_svm = svm.SVC()
clf_svm.fit(features_train, labels_train)
pred_svm = clf_svm.predict(features_test)
score_svm = accuracy_score(labels_test, pred_svm) * 100
print("SVM Accuracy: {:.3f}%".format(score_svm))

# Confusion Matrix for SVM
cm_svm = confusion_matrix(labels_test, pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=[0, 1])
disp_svm.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - SVM')
plt.show()

# Model 3: Random Forest
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(features_train, labels_train)
pred_rf = clf_rf.predict(features_test)
score_rf = accuracy_score(labels_test, pred_rf) * 100
print("Random Forest Accuracy: {:.3f}%".format(score_rf))

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(labels_test, pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=[0, 1])
disp_rf.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Random Forest')
plt.show()

# Model 4: Artificial Neural Network (ANN)
tf.random.set_seed(42)  # For reproducibility

ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, activation='relu', input_shape=(features_train.shape[1],)),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(features_train, labels_train, batch_size=32, epochs=100)

# Evaluate ANN
ann_score = ann.evaluate(features_test, labels_test)
print("ANN Accuracy: {:.3f}%".format(ann_score[1] * 100))

# Confusion Matrix for ANN
pred_ann = (ann.predict(features_test) > 0.5).astype("int32")  # Convert probabilities to binary predictions
cm_ann = confusion_matrix(labels_test, pred_ann)
disp_ann = ConfusionMatrixDisplay(confusion_matrix=cm_ann, display_labels=[0, 1])
disp_ann.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - ANN')
plt.show()
