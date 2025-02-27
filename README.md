# miRNA-Based Disease Prediction with Machine Learning

## Overview
This repository contains the implementation of an optimized machine learning framework for predicting diseases based on microRNA (miRNA) data. The study integrates advanced feature selection, data balancing, and ensemble learning techniques to improve classification accuracy while ensuring model interpretability.

## Features
- **Dimensionality Reduction**: Principal Component Analysis (PCA) for feature selection.
- **Machine Learning Models**: Voting Classifier (Naïve Bayes, SVM, Random Forest) and an Artificial Neural Network (ANN).
- **Explainability**: SHAP and LIME for model interpretation.
- **Performance**: ANN achieves **93% accuracy**, outperforming traditional classifiers.

## Dataset
The dataset is sourced from the **Human microRNA Disease Database (HMDD v3.2)**, a curated repository of experimentally validated miRNA-disease associations.

## Methodology
1. **Data Preprocessing**  
   - Cleaning and normalizing miRNA sequences.  
   - Class balancing using **SMOTE**.  
2. **Feature Engineering**  
   - Extracting miRNA sequence characteristics.  
   - Gene Ontology (GO) term enrichment.  
3. **Model Training**  
   - SVM, Naïve Bayes, Random Forest, and ANN.  
   - Hyperparameter tuning for optimal performance.  
4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-score, and Confusion Matrix.  
5. **Explainability**  
   - SHAP & LIME to analyze feature contributions.  

## Results
- **ANN Model**: Highest accuracy (**93%**), best generalization.  
- **Random Forest**: Balanced accuracy (**91%**) with high interpretability.  
- **SVM & Naïve Bayes**: Effective but less accurate in complex cases.

## Installation & Usage
### Prerequisites
Ensure you have Python and the required dependencies installed:

```sh
pip install numpy pandas scikit-learn shap lime matplotlib seaborn tensorflow keras
```

## Running the Model
# Clone the repository and run the main script:

```sh
git clone https://github.com/yourusername/miRNA-Disease-Prediction.git
```
```sh
cd miRNA-Disease-Prediction
```
```sh
python main.py
```
