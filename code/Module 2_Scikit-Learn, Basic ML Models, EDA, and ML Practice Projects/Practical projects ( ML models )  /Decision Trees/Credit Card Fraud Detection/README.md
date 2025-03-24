# Advanced Credit Card Fraud Detection

![Credit Card Fraud Detection](https://img.shields.io/badge/Project-Credit%20Card%20Fraud%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.x-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📊 Project Overview

This project implements advanced machine learning techniques to detect fraudulent credit card transactions. Using a highly imbalanced dataset (0.17% fraudulent transactions), the model identifies potential fraud with high precision and recall.

### Dataset

The dataset contains transactions made by credit cards, where we have:
- 284,807 transactions 
- 492 fraudulent cases (0.17%)
- 30 input features (V1-V28, Time, Amount)
- Features are PCA transformed for confidentiality

[Dataset Source: Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🔍 Exploratory Analysis Highlights

- **Class Imbalance**: 283,253 normal vs. 473 fraudulent transactions
- **Time Analysis**: Certain hours show higher fraud rates, particularly early morning (1-2 AM, 4 AM)
- **Amount Analysis**: Fraudulent transactions are typically higher in value but mostly remain below $2,000
- **Feature Importance**: V14, V4, V12, and V10 are among the most predictive features

## 🛠️ Methodology

1. **Data Preprocessing**:
    - Feature engineering (time-based features, logarithmic transformations)
    - Feature scaling with RobustScaler
    - SMOTE oversampling to balance the classes

2. **Model Development**:
    - Decision Tree Classifier with hyperparameter tuning
    - Performance metrics focusing on F1-score due to class imbalance

3. **Evaluation**:
    - Confusion matrix analysis
    - ROC-AUC and precision-recall curves
    - Feature importance interpretation

## 📈 Results

The tuned Decision Tree model achieved:
- **Precision**: 0.87 - High confidence in fraud predictions
- **Recall**: 0.89 - Successfully catches most fraud cases
- **F1 Score**: 0.88 - Strong balance between precision and recall
- **ROC AUC**: 0.95 - Excellent discrimination ability

## 🔎 Key Insights

- V14 dominates feature importance (score >0.7)
- Engineered features like V14V17 and V4_squared significantly improve model performance
- Transaction amount alone has limited predictive power for fraud detection
- Model effectively captures time-based patterns of fraudulent activity

## 🚀 Getting Started

### Prerequisites

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
plotly
shap
```

### Technical Setup

1. Make sure you have Python 3.x installed
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Download the dataset from Kaggle and place it in the `data/` directory

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-credit-card-fraud-detection.git
cd advanced-credit-card-fraud-detection

# Install required packages
pip install -r requirements.txt

# Download dataset (requires Kaggle API setup)
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

### Usage

Run the Jupyter notebook to explore the data and see the model in action:

```bash
jupyter notebook main.ipynb
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The dataset was provided by Worldline and the Machine Learning Group at ULB
- Special thanks to the Kaggle community for inspirational baseline models