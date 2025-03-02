import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from imblearn.over_sampling import SMOTE
import shap
import tensorflow as tf
import category_encoders as ce
from scipy.stats import ttest_rel
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


def preprocess_data():
    print("Loading datasets...")
    df_causal = pd.read_excel('data/causal_info.xlsx')
    df_dsw = pd.read_excel('data/dsw.xlsx')

    # Merge datasets
    print("Merging datasets...")
    df = pd.merge(df_causal, df_dsw, how="left", on=["mir", "disease", "pmid"])

    # Handle missing values
    df.dropna(inplace=True)

    # Replace target labels
    print("Replacing target labels...")
    df['causality'] = df['causality'].map({'no': 0, 'yes': 1}).astype(int)

    return df


def feature_engineering(df):
    # Split features and labels
    labels = df['causality']
    features = df.drop(['causality'], axis=1)

    # Identify categorical columns
    categorical_columns = features.select_dtypes(include=['object']).columns

    # Target Encoding for categorical features
    encoder = ce.TargetEncoder(cols=categorical_columns)
    features_encoded = encoder.fit_transform(features, labels)

    # Scale numerical features
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_encoded)

    # Select top features
    selector = SelectKBest(score_func=f_classif, k=50)
    features_selected = selector.fit_transform(features_scaled, labels)

    # Correlation Heatmap (Restored)
    plt.figure(figsize=(15, 12))
    correlation_matrix = pd.DataFrame(features_selected).corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    return features_selected, labels


def train_and_evaluate_models(features, labels):
    # Train-test split
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    features_train_balanced, labels_train_balanced = smote.fit_resample(features_train, labels_train)

    # Define models
    models = {
        'Naive Bayes': GaussianNB(),
        'SVM': svm.SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    # Voting Classifier (Restored)
    voting_clf = VotingClassifier(estimators=list(models.items()), voting='soft')
    models['Voting Classifier'] = voting_clf

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(features_train_balanced, labels_train_balanced)

        # Test predictions
        test_pred = model.predict(features_test)

        print(f"{name} Test Accuracy: {accuracy_score(labels_test, test_pred):.3f}")

    # Confusion Matrix for Voting Classifier (Restored)
    cm = confusion_matrix(labels_test, voting_clf.predict(features_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Voting Classifier')
    plt.tight_layout()
    plt.show()

    # ANN Model (Kept as is)
    ann = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(features_train_balanced.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train ANN
    ann.fit(features_train_balanced, labels_train_balanced, validation_split=0.2, batch_size=32, epochs=20, verbose=1)

    # Evaluate ANN
    ann_score = ann.evaluate(features_test, labels_test)
    print(f"\nANN Test Accuracy: {ann_score[1]:.3f}")

    # SHAP Explainability for Random Forest (Restored)
    print("\nGenerating SHAP explanations for Random Forest...")
    explainer = shap.Explainer(models['Random Forest'], features_test)
    shap_values = explainer(features_test)

    # Create SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

    return ann, features_test


def save_predictions(ann, features_test):
    # Get probability predictions from the ANN model
    ann_prob = ann.predict(features_test)

    # Convert predictions to DataFrame
    df_results = pd.DataFrame({
        "Patient ID": range(1, len(ann_prob) + 1),
        "Cancer Probability (%)": (ann_prob.flatten() * 100),
        "Prediction": ["Yes" if p > 0.5 else "No" for p in ann_prob.flatten()]
    })

    # Filter only "Yes" cases
    df_yes = df_results[df_results["Prediction"] == "Yes"]

    # Print only the patients predicted to have cancer
    print("\nPatients Predicted to Have Cancer:")
    for index, row in df_yes.iterrows():
        print(f"Patient {int(row['Patient ID'])}: Cancer Probability: {row['Cancer Probability (%)']:.2f}% - Yes")

    # Define CSV file path
    file_path = "cancer_positive_patients.csv"

    # Check if file exists, then append only new cases
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_combined = pd.concat([df_existing, df_yes]).drop_duplicates()
        df_combined.to_csv(file_path, index=False)
    else:
        df_yes.to_csv(file_path, index=False)

    print(f"\nUpdated {file_path} with new cancer-positive patients.")


def main():
    # Full pipeline
    df = preprocess_data()
    features, labels = feature_engineering(df)
    ann, features_test = train_and_evaluate_models(features, labels)
    save_predictions(ann, features_test)


if __name__ == "__main__":
    main()
