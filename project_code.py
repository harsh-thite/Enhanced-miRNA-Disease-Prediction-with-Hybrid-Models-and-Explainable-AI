import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from imblearn.over_sampling import SMOTE
import shap
import tensorflow as tf
import category_encoders as ce
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


def preprocess_data():
    print("Loading datasets...")
    df_causal = pd.read_excel('data/causal_info.xlsx')
    df_dsw = pd.read_excel('data/dsw.xlsx')

    print("Merging datasets...")
    df = pd.merge(df_causal, df_dsw, how="left", on=["mir", "disease", "pmid"])
    df.dropna(inplace=True)

    print("Replacing target labels...")
    df['causality'] = df['causality'].map({'no': 0, 'yes': 1}).astype(int)

    return df


def feature_engineering(df):
    labels = df['causality']
    features = df.drop(['causality'], axis=1)

    categorical_columns = features.select_dtypes(include=['object']).columns
    encoder = ce.TargetEncoder(cols=categorical_columns)
    features_encoded = encoder.fit_transform(features, labels)

    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_encoded)

    selector = SelectKBest(score_func=f_classif, k=20)  # Reduce overfitting
    features_selected = selector.fit_transform(features_scaled, labels)

    # Feature Correlation Heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = pd.DataFrame(features_selected).corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()

    return features_selected, labels


def train_and_evaluate_models(features, labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(features, labels):
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels.iloc[train_index], labels.iloc[test_index]

    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    features_train_balanced, labels_train_balanced = smote.fit_resample(features_train, labels_train)

    # Class Distribution Graph (Before & After SMOTE)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Original Class 0", "Original Class 1"], y=[sum(labels_train == 0), sum(labels_train == 1)],
                color='blue', label="Before SMOTE")
    sns.barplot(x=["Original Class 0", "Original Class 1"],
                y=[sum(labels_train_balanced == 0), sum(labels_train_balanced == 1)], color='red', alpha=0.6,
                label="After SMOTE")
    plt.title("Class Distribution Before & After SMOTE")
    plt.legend()
    plt.show()

    models = {
        'Naive Bayes': GaussianNB(),
        'SVM': svm.SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    voting_clf = VotingClassifier(estimators=list(models.items()), voting='soft')
    models['Voting Classifier'] = voting_clf

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(features_train_balanced, labels_train_balanced)
        test_pred = model.predict(features_test)

        print(f"{name} Test Accuracy: {accuracy_score(labels_test, test_pred):.3f}")
        print(f"\n{name} Classification Report:\n")
        print(classification_report(labels_test, test_pred))

    # Confusion Matrix for Voting Classifier
    cm = confusion_matrix(labels_test, voting_clf.predict(features_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Voting Classifier')
    plt.show()

    # ANN Model
    ann = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(features_train_balanced.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    ann.summary()  # ANN Training Summary

    history = ann.fit(features_train_balanced, labels_train_balanced, validation_split=0.2, batch_size=32, epochs=20,
                      verbose=1)
    ann_score = ann.evaluate(features_test, labels_test)

    print(f"\nANN Test Accuracy: {ann_score[1]:.3f}")

    ann_pred_prob = ann.predict(features_test).flatten()
    ann_pred = (ann_pred_prob > 0.5).astype("int32")

    # Confusion Matrix for ANN Model
    cm_ann = confusion_matrix(labels_test, ann_pred)
    disp_ann = ConfusionMatrixDisplay(confusion_matrix=cm_ann, display_labels=[0, 1])
    plt.figure(figsize=(8, 6))
    disp_ann.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - ANN')
    plt.show()

    # ROC Curve for ANN
    fpr, tpr, _ = roc_curve(labels_test, ann_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ANN AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve - ANN")
    plt.legend()
    plt.show()

    # Precision-Recall Curve for ANN
    precision, recall, _ = precision_recall_curve(labels_test, ann_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.title("Precision-Recall Curve - ANN")
    plt.legend()
    plt.show()

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
    ann_prob = ann.predict(features_test)

    df_results = pd.DataFrame({
        "Patient ID": range(1, len(ann_prob) + 1),
        "Cancer Probability (%)": (ann_prob.flatten() * 100),
        "Prediction": ["Yes" if p > 0.5 else "No" for p in ann_prob.flatten()]
    })

    # Print "Yes" & "No" Predictions
    print("\nAll Patient Predictions:")
    for index, row in df_results.iterrows():
        print(f"Patient {int(row['Patient ID'])}: Cancer Probability: {row['Cancer Probability (%)']:.2f}% - {row['Prediction']}")

    # Save Only "Yes" Cases for CSV
    df_yes = df_results[df_results["Prediction"] == "Yes"]

    file_path = "cancer_positive_patients.csv"
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_combined = pd.concat([df_existing, df_yes]).drop_duplicates()
        df_combined.to_csv(file_path, index=False)
    else:
        df_yes.to_csv(file_path, index=False)

    print(f"\nUpdated {file_path} with new cancer-positive patients.")

def main():
    df = preprocess_data()
    features, labels = feature_engineering(df)
    ann, features_test = train_and_evaluate_models(features, labels)
    save_predictions(ann, features_test)  # Saving Predictions


if __name__ == "__main__":
    main()