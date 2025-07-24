# Credit Default Prediction Model

This project implements a logistic regression model to predict credit defaults using a comprehensive dataset of loan applications.

## Files Overview

- `credit_default_modeling.ipynb` - Main Jupyter notebook with complete analysis and modeling
- `credit_risk_dataset.csv` - Dataset containing loan and borrower information
- `requirements.txt` - Required Python packages

## Setup Instructions

1. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook credit_default_modeling.ipynb
   ```

3. **Run the Analysis**
   - Execute cells sequentially from top to bottom
   - The notebook will automatically load the dataset and perform all analysis steps

## Notebook Contents

### 1. Data Loading and Overview
- Loads the credit risk dataset
- Provides basic statistics and data structure information

### 2. Exploratory Data Analysis
- Analyzes missing values and data quality
- Visualizes target variable distribution
- Examines default rates across different categories
- Creates correlation analysis

### 3. Data Preprocessing
- Handles missing values using intelligent imputation
- Removes extreme outliers and data quality issues
- Encodes categorical variables
- Engineers new features (age groups, income quartiles, etc.)

### 4. Model Building
- Splits data into training and testing sets
- Scales features using StandardScaler
- Builds baseline logistic regression model
- Performs hyperparameter tuning with GridSearchCV

### 5. Model Evaluation
- Comprehensive evaluation with multiple metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC and ROC curves
  - Confusion matrices
  - Classification reports

### 6. Feature Importance Analysis
- Analyzes logistic regression coefficients
- Identifies top risk factors and protective factors
- Visualizes feature importance

### 7. Predictions and Business Insights
- Categorizes loans by risk level
- Provides business recommendations
- Calculates potential financial impact

## Key Features

### Model Performance Metrics
- **ROC-AUC**: Measures discriminative ability
- **Precision**: Accuracy of default predictions
- **Recall**: Coverage of actual defaults
- **F1-Score**: Balanced performance metric

### Risk Factors Identified
1. **Loan Grade** - Primary risk indicator
2. **Previous Default History** - Strong predictor
3. **Loan-to-Income Ratio** - Financial stress indicator
4. **Interest Rate** - Risk pricing reflection
5. **Employment Length** - Job stability factor

### Business Value
- **Risk Assessment**: Automated loan risk scoring
- **Decision Support**: Data-driven loan approval
- **Pricing Optimization**: Risk-based interest rates
- **Loss Prevention**: Early default identification

## Model Output

The notebook generates several artifacts:
- `credit_default_model.pkl` - Trained logistic regression model
- `feature_scaler.pkl` - Feature scaling object
- `label_encoders.pkl` - Categorical encoding mappings

## Usage for New Predictions

```python
import joblib
import pandas as pd

# Load trained model and preprocessors
model = joblib.load('credit_default_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
encoders = joblib.load('label_encoders.pkl')

# Prepare new data (follow same preprocessing steps)
# Apply scaling and encoding
# Make predictions
risk_probability = model.predict_proba(new_data_scaled)[:, 1]
```

## Dataset Information

**Total Records**: ~32,000 loan applications  
**Features**: 12 key variables covering:
- **Demographics**: Age, income, home ownership, employment
- **Loan Details**: Amount, grade, interest rate, purpose
- **Risk Indicators**: Default history, credit history length

**Target Variable**: `loan_status` (0 = Good, 1 = Default)

## Model Interpretation

### High-Risk Indicators
- Lower loan grades (D, E, F, G)
- Previous default history
- High loan-to-income ratios (>40%)
- Certain loan purposes (venture, medical)

### Protective Factors
- Higher income levels
- Longer employment history
- Home ownership
- Lower loan-to-income ratios

## Next Steps

1. **Production Deployment**: Implement model in loan processing system
2. **Model Monitoring**: Track performance over time
3. **Retraining Pipeline**: Regular model updates with new data
4. **Ensemble Methods**: Combine with other algorithms for improved performance
5. **Explainability**: Develop SHAP or LIME explanations for regulatory compliance

## Requirements

- Python 3.7+
- Jupyter Notebook
- See `requirements.txt` for complete package list

## Support

For questions or issues, please review the notebook comments and documentation within each cell. 
