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

    # Filter specific diseases
    diseases = [
        'Alopecia', 'Alzheimer Disease', 'Asthma', 'Brain Neoplasms', 
        'Cataract', 'Down Syndrome', 'Endometriosis', 'Gastric Neoplasms', 
        'Glaucoma', 'Heart Failure', 'Hypertension', 'Ischemic Stroke', 
        'Kidney Failure', 'Leukemia', 'Lung Neoplasms', 'Melanoma', 
        'Pancreatic Neoplasms', 'Parkinson Disease', 'Schizophrenia', 'Stroke'
    ]
    df = df[df['disease'].isin(diseases)]

    # Sample data if too large
    if len(df) > 5000:
        print(f"Dataset has {len(df)} rows. Sampling 5000 rows...")
        df = df.sample(n=5000, random_state=42)
    else:
        print(f"Dataset has {len(df)} rows. No sampling required.")

    return df

def feature_engineering(df):
    # Split features and labels
    labels = df['causality']
    features = df.drop(['causality'], axis=1)

    # Identify categorical and numerical columns
    categorical_columns = features.select_dtypes(include=['object']).columns
    numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns

    # Target Encoding for categorical features
    encoder = ce.TargetEncoder(cols=categorical_columns)
    features_encoded = encoder.fit_transform(features, labels)

    # Scale numerical features
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_encoded)

    # Select top features before SMOTE
    selector = SelectKBest(score_func=f_classif, k=50)
    features_selected = selector.fit_transform(features_scaled, labels)

    # Correlation Heatmap
    plt.figure(figsize=(15, 12))
    correlation_matrix = pd.DataFrame(features_selected).corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    print(f"Selected features shape: {features_selected.shape}")
    return features_selected, labels

def train_and_evaluate_models(features, labels):
    # Train-test split
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Apply SMOTE after train-test split
    smote = SMOTE(random_state=42)
    features_train_balanced, labels_train_balanced = smote.fit_resample(features_train, labels_train)

    print("SMOTE applied. New training label distribution:")
    print(labels_train_balanced.value_counts())

    # Define models
    models = {
        'Naive Bayes': GaussianNB(),
        'SVM': svm.SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    # Voting Classifier
    voting_clf = VotingClassifier(
        estimators=list(models.items()), voting='soft'
    )
    models['Voting Classifier'] = voting_clf

    # Training and evaluation results storage
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(features_train_balanced, labels_train_balanced)
        
        # Predictions and metrics
        train_pred = model.predict(features_train_balanced)
        test_pred = model.predict(features_test)
        
        results[name] = {
            'train_time': time.time() - start_time,
            'train_accuracy': accuracy_score(labels_train_balanced, train_pred),
            'test_accuracy': accuracy_score(labels_test, test_pred),
            'classification_report': classification_report(labels_test, test_pred)
        }
        
        print(f"{name} training completed in {results[name]['train_time']:.2f} seconds.")
        print(f"Train Accuracy: {results[name]['train_accuracy']:.3f}")
        print(f"Test Accuracy: {results[name]['test_accuracy']:.3f}")
        print("\nClassification Report:")
        print(results[name]['classification_report'])

    # Confusion Matrix for Voting Classifier
    cm = confusion_matrix(labels_test, voting_clf.predict(features_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Voting Classifier')
    plt.tight_layout()
    plt.show()

    # Statistical Comparison
    model_names = list(models.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            m1_preds = models[model_names[i]].predict(features_test)
            m2_preds = models[model_names[j]].predict(features_test)
            t_stat, p_val = ttest_rel(m1_preds, m2_preds)
            print(f"\nT-Test between {model_names[i]} and {model_names[j]}:")
            print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.3f}")

    # ANN Model
    ann = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(features_train_balanced.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train ANN
    history = ann.fit(
        features_train_balanced, labels_train_balanced, 
        validation_split=0.2, 
        batch_size=32, 
        epochs=20, 
        verbose=1
    )

    # Evaluate ANN
    ann_score = ann.evaluate(features_test, labels_test)
    print(f"\nANN Accuracy: {ann_score[1]:.3f}")

    # SHAP Explainability for Random Forest
    print("\nGenerating SHAP explanations for Random Forest...")
    explainer = shap.Explainer(models['Random Forest'], features_test)
    shap_values = explainer(features_test)

    # Create SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

    return results

def main():
    # Full pipeline
    df = preprocess_data()
    features, labels = feature_engineering(df)
    results = train_and_evaluate_models(features, labels)

if __name__ == "__main__":
    main()
