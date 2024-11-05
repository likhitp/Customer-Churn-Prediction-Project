# Customer Churn Prediction Project

---

## Table of Contents
- [Project Overview](#project-overview)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering & Selection](#feature-engineering-and-selection)
- [Data Preprocessing Decisions](#data-preprocessing-decisions)
- [Model Development](#model-development)
- [Results & Evaluation](#results-and-evaluation)
- [Technical Implementation](#technical-implementation)
- [Future Improvements](#future-improvements)

---

## 1. Project Overview

### Business Context
This project aims to address a significant challenge in the banking sector: customer churn prediction. Retaining customers is more cost-effective than acquiring new ones, making it crucial for profitability.


## Installation

```bash
# Clone the repository
git clone https://github.com/likhitp/Customer-Churn-Prediction-Project.git

# Navigate to project directory
cd bank-churners

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the preprocessing pipeline
python src/main.py

# Train the models
python src/model_training.py

# Generate visualizations
python src/visualizations.py
```

## Project Structure

```
BANK_CHURNERS/
├── data/                    # Dataset storage
│   └── BankChurners.csv
├── models/                  # Trained models and scalers
│   └── scalers/
├── notebooks/               # Jupyter notebooks for exploration
│   ├── exploratory_analysis.ipynb
│   └── model_development.ipynb
├── src/                     # Source code
│   ├── main.py              # Main preprocessing pipeline
│   ├── model_training.py    # Model training scripts
│   └── visualizations.py    # Visualization generation
├── visualizations/          # Generated plots and figures
├── README.md
└── requirements.txt
```

## Technologies Used

- Python 3.8+
- scikit-learn
- XGBoost
- pandas
- numpy
- matplotlib
- seaborn

### Key Business Objectives
- **Demographic Analysis**: Identify high-risk customer segments and patterns in gender, marital status, education, and income.
- **Behavioral Pattern Analysis**: Examine transaction behaviors, credit usage, and engagement metrics for early churn warning signs.
- **Predictive Modeling**: Develop a robust model to predict churn, empowering proactive retention strategies.

#### Initial Findings
- **Transaction Behavior**: Churned customers exhibit lower transaction amounts ($3,095 vs $4,654).
- **Engagement Metrics**: Lower transaction frequency among churned customers (45 vs 69 transactions).
- **Credit Usage**: Lower revolving balance ($673 vs $1,257) and utilization (16% vs 30%).

---

## 2. Exploratory Data Analysis

### 2.1 Data Understanding
The dataset includes customer demographics, transaction history, and credit-related information. The target variable, `Attrition_Flag`, represents customer attrition, framing this as a binary classification problem with both categorical and numerical features.

### 2.2 Key Visualization Insights
#### Transaction and Credit Patterns
1. **Box Plot Analysis**:
   - Identified significant outliers in `Total_Trans_Amt`.
   - Observed skewed distribution in `Credit_Limit`.

     leading to outlier treatment.

     ![boxplot_std](https://github.com/user-attachments/assets/99669eb7-b954-4fe4-bf65-f07e6166c6eb)

   <suggestions: Consider including a box plot to visually convey these findings.>

1. **Correlation Analysis**:
   - Noted strong correlation between `Credit_Limit` and `Avg_Open_To_Buy`.
   - Moderate correlation observed between `Total_Trans_Amt` and `Total_Trans_Ct`.
  
   ![correlation_heatmap](https://github.com/user-attachments/assets/877ec85a-de7b-45c6-8f1e-00abc76c2787)



  

2. **Distribution Analysis**:
   - Discovered skewed distributions in financial features.
   - Clear separation between churners and non-churners in transaction patterns.
  
![dist_0](https://github.com/user-attachments/assets/471572b7-528b-4173-bffa-faa9b0ab8ff9)

![pairplot_derived](https://github.com/user-attachments/assets/3ed7f52e-254d-4ef2-a49a-543a7c8a7574)


---

### 2.3 Critical Patterns Identified

In the exploratory analysis, several critical patterns emerged that are highly indicative of customer churn risk. These patterns fall under three main categories: transaction patterns, credit usage, and customer engagement. These indicators highlight differences in customer behaviors, allowing for actionable insights that can guide proactive retention strategies.

#### Customer Behavior Indicators

##### **Transaction Patterns**
- **Lower Transaction Amounts**: Customers who are at a higher risk of churning tend to have lower transaction amounts. This trend suggests a potential reduction in engagement and usage of the bank’s services among churn-prone customers, potentially signaling dissatisfaction or loss of interest.
- **Reduced Transaction Frequency**: Another key early warning sign for churn is a lower frequency of transactions. Customers who make fewer transactions are less likely to have regular interactions with the bank, which could indicate a lack of perceived value in the bank’s offerings or a shift to using other financial services.

##### **Credit Usage**
- **Lower Revolving Balance**: Customers who churn often have lower revolving balances. This may imply that they are less reliant on the bank's credit facilities or could be moving their balances to competing services. By identifying such patterns, the bank can consider targeted offers to re-engage these customers.
- **Reduced Credit Utilization**: Prior to churning, customers tend to exhibit a decline in their credit utilization ratio. Lower utilization could be a result of the customer reducing their dependence on credit products or potentially shifting to alternative financial providers.

##### **Customer Engagement**
- **Higher Inactive Months**: Customers who show higher counts of inactive months are significantly more likely to churn. Extended inactivity could reflect disinterest or low satisfaction with the services provided, making it an essential factor for customer engagement teams to monitor closely.
- **Lower Relationship Count**: Customers with fewer product relationships (e.g., checking, savings, loans) are at higher risk for churn. The relationship count is a measure of how deeply integrated a customer is with the bank, so fewer connections imply a weaker bond and, thus, a higher risk of attrition.

![transaction_patterns](https://github.com/user-attachments/assets/d448dd83-e6c7-4c27-b379-e0932238f460)

![engagement_patterns](https://github.com/user-attachments/assets/6eea28fb-d855-4c69-9882-69acb550ca9b)

---

### 2.4 Feature Importance Discovery

The analysis identified the most significant predictive features influencing churn, revealing valuable insights into which factors drive customer retention. Feature importance was determined using model-based methods, specifically evaluating the impact of each feature on churn predictions. Here’s a deeper look at the top features:

#### Top Predictive Features

- **Total_Trans_Amt (21.5%)**: This feature, representing the total transaction amount over a specified period, emerged as the most critical predictor. Higher transaction amounts correlate with retention, suggesting that customers who frequently use their accounts tend to remain loyal.
- **Total_Trans_Ct (21.2%)**: The total count of transactions is nearly as influential as the transaction amount. High transaction counts indicate strong engagement and are a positive retention factor, as these customers interact with the bank’s services regularly.
- **Total_Revolving_Bal (12.0%)**: Revolving balance, representing credit usage, is another significant predictor. Customers with lower revolving balances are more likely to churn, potentially due to reduced credit dependency. Monitoring changes in revolving balances could help identify customers at risk of attrition.
- **Avg_Utilization_Ratio (7.9%)**: Average utilization ratio reflects the extent to which customers use their available credit. Customers with a low utilization ratio are more likely to churn, suggesting a decreased reliance on the bank’s credit facilities. Proactive strategies can target these users to encourage higher credit engagement.
- **Total_Relationship_Count (7.5%)**: This metric indicates the number of products a customer has with the bank. A higher relationship count signifies a deeper connection with the bank, which is associated with lower churn risk. Encouraging customers to open multiple accounts or products can enhance their engagement and loyalty.


![feature_importance](https://github.com/user-attachments/assets/66c4db92-9ac1-436d-a772-6da40d21376f)


---

## 3. Feature Engineering & Selection

### Engineering Decisions

To enhance predictive power, several derived features were created to capture complex customer behaviors and improve model interpretability. Key derived features include:

- **Trans_to_Credit_Ratio**: Measures spending capacity utilization by comparing transaction amounts to available credit.
- **Avg_Transaction_per_Contact**: Represents transaction efficiency by dividing total transactions by the number of contacts, indicating customer responsiveness.
- **Inactive_Contact_Ratio**: Reflects engagement decline, calculated as the ratio of inactive months to contacts made.
- **Relationship_Tenure**: Captures customer loyalty based on months on book and relationship depth.

### Feature Selection

- **Removed**: `Avg_Open_To_Buy` was removed due to its 99% correlation with `Credit_Limit`, indicating redundancy.
- **Retained**: Transaction pairs (e.g., `Total_Trans_Amt`, `Total_Trans_Ct`) were kept due to their individual predictive power despite correlation.
- **Prioritized**: Features demonstrating clear separability between churners and non-churners were prioritized, contributing to model performance improvements.


---

## 4. Data Preprocessing Pipeline

### Transformation Strategy

A robust preprocessing pipeline was implemented to prepare data for modeling. The key steps include:

- **Outlier Handling**:
  - Winsorization at the 1st and 99th percentiles was applied to monetary features to limit the impact of extreme values.
  - Log transformation was applied to skewed distributions, particularly for financial features, to achieve normality.

- **Scaling**:
  - `StandardScaler` was used for monetary features such as `Credit_Limit` and `Total_Trans_Amt`.
  - `MinMaxScaler` was applied to other numerical features to ensure consistent scaling across all data.

- **Missing Values**:
  - For numerical features, mean imputation was used to replace missing values.
  - For categorical features, mode imputation was implemented.
 

 
![transform_impact_Total_Trans_Amt](https://github.com/user-attachments/assets/3b81cc4d-393a-4e5b-a03d-4fde2d5a6b5d)



---

## 5. Model Development

### Model Selection

Three machine learning models were tested for this project, each with unique advantages:

1. **Logistic Regression (Baseline)**:
   - **ROC AUC**: 0.9420
   - Simple and interpretable, providing baseline performance with reasonable accuracy.

2. **Random Forest**:
   - **ROC AUC**: 0.9825
   - Effectively handles feature interactions, showing an improvement in predictive accuracy.

3. **XGBoost (Champion)**:
   - **ROC AUC**: 0.9919
   - Best overall performance, particularly with a precision of 0.92 on churned customers, making it the champion model for identifying high-risk customers.
  
   ![precision_recall_curves](https://github.com/user-attachments/assets/352d4c8e-6006-4c7a-9b64-8e670de70af6)
![roc_curves](https://github.com/user-attachments/assets/b0bfc618-f0b9-4a7c-b11d-9db5f45e311e)


### Class Imbalance

The original class ratio was heavily imbalanced, with 84% non-churners and 16% churners. To address this:

- **Solution**: SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the dataset.
- **Result**: Balanced training set without information loss, enabling more reliable model predictions.




---

## 6. Results & Business Impact

### Model Performance

The final model achieved high precision and recall, proving effective in identifying customers at risk of churn:

- **Precision**: 92% in identifying churners
- **Recall**: 87% for catching potential churners
- **False Positives**: Only 15 false positives per 1,328 predictions, maintaining operational efficiency.

### Business Recommendations

Based on the model’s insights, several strategies are recommended to mitigate churn:

1. **Transaction Monitoring**:
   - Implement an early warning system to detect decreased activity levels.
   - Track changes in both transaction amount and frequency for timely intervention.

2. **Credit Usage**:
   - Regularly monitor trends in revolving balance and credit utilization rates.
   - Identify and address shifts in credit usage that may precede churn.

3. **Customer Engagement**:
   - Prioritize relationship-building activities, focusing on increasing product adoption and engagement.
   - Establish intervention protocols to act at the first sign of declining engagement.

<Todo: Provide a sample visualization or dashboard layout for the recommended early warning system, illustrating how transaction and credit usage metrics can be monitored.>

---

## 7. Future Improvements

### Technical Enhancements

1. **Model Optimization**:
   - Implement automated threshold tuning for optimal decision boundaries.
   - Add cross-validation steps for model robustness to avoid overfitting.

2. **Production Readiness**:
   - Set up feature drift monitoring to track data changes over time.
   - Develop an automated retraining pipeline for continuous model updates.
   - Create an API endpoint to enable real-time churn predictions for operational teams.

### Business Extensions

1. **Customer Segmentation**:
   - Use the model insights to develop segment-specific retention strategies.
   - Create personalized intervention programs based on identified risk factors.

2. **Real-time Monitoring**:
   - Integrate a live transaction pattern analysis to enable real-time churn risk assessment.
   - Implement immediate risk score updates for high-frequency customer engagement.

<Todo: Add diagrams illustrating the envisioned real-time monitoring architecture and segmentation strategies. Visual representations can help stakeholders visualize these potential future improvements.>

---

This project showcases my skills in Python, critical thinking in data processing, and analytical approaches to customer churn prediction. With clear actionable insights, it provides a foundation for proactive retention strategies in the banking sector.
