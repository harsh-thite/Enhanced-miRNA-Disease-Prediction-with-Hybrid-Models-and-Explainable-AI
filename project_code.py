import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score
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
    
    # Print class distribution to check imbalance
    print(f"Class distribution - 0: {sum(df['causality'] == 0)}, 1: {sum(df['causality'] == 1)}")
    print(f"Imbalance ratio: 1:{sum(df['causality'] == 0)/sum(df['causality'] == 1):.2f}")

    return df


def split_data(df):
    """Split data before any preprocessing to prevent data leakage"""
    labels = df['causality']
    features = df.drop(['causality'], axis=1)
    
    # First split the data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(features, labels):
        features_train_raw = features.iloc[train_index]
        features_test_raw = features.iloc[test_index]
        labels_train = labels.iloc[train_index]
        labels_test = labels.iloc[test_index]
    
    return features_train_raw, features_test_raw, labels_train, labels_test


def feature_engineering(features_train_raw, features_test_raw, labels_train):
    """Process features correctly with no data leakage"""
    # Handle categorical features
    categorical_columns = features_train_raw.select_dtypes(include=['object']).columns
    encoder = ce.TargetEncoder(cols=categorical_columns)
    features_train_encoded = encoder.fit_transform(features_train_raw, labels_train)
    # Important: Use the same encoder for test data
    features_test_encoded = encoder.transform(features_test_raw)
    
    # Scale features
    scaler = RobustScaler()
    features_train_scaled = scaler.fit_transform(features_train_encoded)
    # Important: Use the same scaler for test data
    features_test_scaled = scaler.transform(features_test_encoded)
    
    # Feature selection - fit only on training data
    selector = SelectKBest(score_func=f_classif, k=20)
    features_train_selected = selector.fit_transform(features_train_scaled, labels_train)
    # Apply the same transformation to test data
    features_test_selected = selector.transform(features_test_scaled)
    
    # Show correlation matrix for selected features
    if features_train_selected.shape[1] <= 20:  # Only if the number of features is reasonable
        plt.figure(figsize=(12, 8))
        correlation_matrix = pd.DataFrame(features_train_selected).corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
        plt.title("Feature Correlation Heatmap (Training Data Only)")
        plt.show()
    
    return features_train_selected, features_test_selected


def train_and_evaluate_models(features_train, features_test, labels_train, labels_test):
    # Check class imbalance before SMOTE
    unique, counts = np.unique(labels_train, return_counts=True)
    print(f"Original training class distribution: {dict(zip(unique, counts))}")
    
    # Apply SMOTE for balancing
    smote = SMOTE(sampling_strategy=0.8, random_state=42)  # Less aggressive balancing (0.8 instead of 0.5)
    features_train_balanced, labels_train_balanced = smote.fit_resample(features_train, labels_train)
    
    # Check class distribution after SMOTE
    unique_balanced, counts_balanced = np.unique(labels_train_balanced, return_counts=True)
    print(f"Balanced training class distribution: {dict(zip(unique_balanced, counts_balanced))}")
    
    # Visualize class balance
    plt.figure(figsize=(8, 5))
    sns.barplot(x=["Original Class 0", "Original Class 1"], 
                y=[sum(labels_train == 0), sum(labels_train == 1)],
                color='blue', label="Before SMOTE")
    sns.barplot(x=["Balanced Class 0", "Balanced Class 1"],
                y=[sum(labels_train_balanced == 0), sum(labels_train_balanced == 1)], 
                color='red', alpha=0.6, label="After SMOTE")
    plt.title("Class Distribution Before & After SMOTE")
    plt.legend()
    plt.show()
    
    # Also show test set distribution
    plt.figure(figsize=(6, 4))
    test_unique, test_counts = np.unique(labels_test, return_counts=True)
    sns.barplot(x=[f"Class {i}" for i in test_unique], y=test_counts)
    plt.title("Test Set Class Distribution")
    plt.show()

    # Initialize models with more appropriate parameters
    models = {
        'Naive Bayes': GaussianNB(),
        'SVM': svm.SVC(probability=True, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                               class_weight='balanced', random_state=42)
    }

    voting_clf = VotingClassifier(estimators=list(models.items()), voting='soft')
    models['Voting Classifier'] = voting_clf

    metrics_table = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(features_train_balanced, labels_train_balanced)
        
        # Make predictions
        test_pred = model.predict(features_test)
        
        # Calculate metrics
        acc = accuracy_score(labels_test, test_pred)
        prec = precision_score(labels_test, test_pred)
        rec = recall_score(labels_test, test_pred)
        f1 = f1_score(labels_test, test_pred)
        
        metrics_table.append([name, acc, prec, rec, f1])

        print(f"{name} Test Accuracy: {acc:.3f}")
        print(f"{name} Test F1-Score: {f1:.3f}")
        print(f"\n{name} Classification Report:\n")
        print(classification_report(labels_test, test_pred))

        # Plot confusion matrix for each model
        cm = confusion_matrix(labels_test, test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        plt.figure(figsize=(6, 5))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {name}')
        plt.show()

    # Neural Network with a more appropriate architecture
    ann = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(features_train_balanced.shape[1],)),
        tf.keras.layers.Dropout(0.4),  # Increase dropout to prevent overfitting
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Use class weights to handle imbalance
    class_weight = {0: 1.0, 1: sum(labels_train == 0) / sum(labels_train == 1)}
    print(f"ANN class weights: {class_weight}")
    
    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                loss='binary_crossentropy', 
                metrics=['accuracy'])

    ann.summary()

    # Train with validation split and early stopping
    history = ann.fit(
        features_train_balanced, labels_train_balanced, 
        validation_split=0.2, 
        class_weight=class_weight,
        batch_size=32, 
        epochs=50,  # We'll use early stopping
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Evaluate ANN
    ann_score = ann.evaluate(features_test, labels_test)
    ann_pred_prob = ann.predict(features_test).flatten()
    ann_pred = (ann_pred_prob > 0.5).astype("int32")

    acc = accuracy_score(labels_test, ann_pred)
    prec = precision_score(labels_test, ann_pred)
    rec = recall_score(labels_test, ann_pred)
    f1 = f1_score(labels_test, ann_pred)
    metrics_table.append(["ANN", acc, prec, rec, f1])

    print(f"\nANN Test Accuracy: {ann_score[1]:.3f}")
    print(f"ANN Classification Report:\n")
    print(classification_report(labels_test, ann_pred))

    cm_ann = confusion_matrix(labels_test, ann_pred)
    disp_ann = ConfusionMatrixDisplay(confusion_matrix=cm_ann, display_labels=[0, 1])
    plt.figure(figsize=(6, 5))
    disp_ann.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - ANN')
    plt.show()

    # ROC curve
    plt.figure(figsize=(8, 6))
    
    # Add ROC curves for each model
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            model_probs = model.predict_proba(features_test)[:, 1]
        else:
            model_probs = model.predict(features_test)
            
        fpr, tpr, _ = roc_curve(labels_test, model_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    
    # Add ANN ROC curve
    fpr_ann, tpr_ann, _ = roc_curve(labels_test, ann_pred_prob)
    roc_auc_ann = auc(fpr_ann, tpr_ann)
    plt.plot(fpr_ann, tpr_ann, label=f"ANN (AUC = {roc_auc_ann:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curves for All Models")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Precision-Recall curve for ANN
    precision_vals, recall_vals, _ = precision_recall_curve(labels_test, ann_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve")
    plt.title("Precision-Recall Curve - ANN")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # SHAP explanations for Random Forest
    print("\nGenerating SHAP explanations for Random Forest...")
    rf_model = models['Random Forest']
    
    # Create a small sample if data is large
    max_display = min(20, features_test.shape[0])
    sample_indices = np.random.choice(features_test.shape[0], max_display, replace=False)
    sample_test = features_test[sample_indices]
    
    explainer = shap.Explainer(rf_model, sample_test)
    shap_values = explainer(sample_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

    # Model Performance Comparison Table with F1 score
    df_metrics = pd.DataFrame(metrics_table, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    print("\nModel Performance Summary:")
    print(df_metrics)

    # Bar chart for all metrics
    ax = df_metrics.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
        kind="bar", figsize=(12, 6)
    )
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="lower right")
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8)
        
    plt.tight_layout()
    plt.show()

    return ann, features_test, df_metrics


def save_predictions(ann, features_test, labels_test):
    ann_prob = ann.predict(features_test)

    df_results = pd.DataFrame({
        "Patient ID": range(1, len(ann_prob) + 1),
        "Cancer Probability (%)": (ann_prob.flatten() * 100),
        "Prediction": ["Yes" if p > 0.5 else "No" for p in ann_prob.flatten()],
        "True Label": labels_test.reset_index(drop=True)
    })

    print("\nPredictions Summary (First 10 patients):")
    print(df_results.head(10))
    
    # Calculate metrics
    correct = sum(df_results["Prediction"].map({"Yes": 1, "No": 0}) == df_results["True Label"])
    accuracy = correct / len(df_results)
    print(f"\nOverall accuracy: {accuracy:.2f} ({correct} correct out of {len(df_results)})")

    # Save cancer-positive predictions
    df_yes = df_results[df_results["Prediction"] == "Yes"]
    file_path = "cancer_positive_patients.csv"
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_combined = pd.concat([df_existing, df_yes]).drop_duplicates()
        df_combined.to_csv(file_path, index=False)
    else:
        df_yes.to_csv(file_path, index=False)

    print(f"\nUpdated {file_path} with {len(df_yes)} new cancer-positive patients.")
    
    # Save all predictions
    df_results.to_csv("all_predictions.csv", index=False)
    print("All predictions saved to 'all_predictions.csv'")


def main():
    df = preprocess_data()
    
    # Split data before preprocessing
    features_train_raw, features_test_raw, labels_train, labels_test = split_data(df)
    
    # Process features correctly with no data leakage
    features_train, features_test = feature_engineering(features_train_raw, features_test_raw, labels_train)
    
    # Train and evaluate models
    ann, features_test, metrics_df = train_and_evaluate_models(features_train, features_test, labels_train, labels_test)
    
    # Save predictions with true labels for analysis
    save_predictions(ann, features_test, labels_test)
    
    # Save metrics table
    metrics_df.to_csv("model_performance_metrics.csv", index=False)
    print("Model performance metrics saved to 'model_performance_metrics.csv'")


if __name__ == "__main__":
    main()

