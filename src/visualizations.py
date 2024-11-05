import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create directory for new visualizations
if not os.path.exists('visualizations/patterns'):
    os.makedirs('visualizations/patterns')
if not os.path.exists('visualizations/features'):
    os.makedirs('visualizations/features')
if not os.path.exists('visualizations/preprocessing'):
    os.makedirs('visualizations/preprocessing')

# 1. Transaction and Credit Usage Patterns
def plot_transaction_patterns(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Transaction Amount Distribution
    sns.boxplot(x='Attrition_Flag', y='Total_Trans_Amt', data=df, ax=axes[0,0])
    axes[0,0].set_title('Transaction Amounts by Churn Status')
    
    # Transaction Frequency
    sns.histplot(data=df, x='Total_Trans_Ct', hue='Attrition_Flag', 
                multiple="dodge", shrink=.8, ax=axes[0,1])
    axes[0,1].set_title('Transaction Frequency Distribution')
    
    # Credit Usage
    sns.boxplot(x='Attrition_Flag', y='Credit_Limit', data=df, ax=axes[1,0])
    axes[1,0].set_title('Credit Limit Distribution')
    
    # Revolving Balance
    sns.boxplot(x='Attrition_Flag', y='Total_Revolving_Bal', data=df, ax=axes[1,1])
    axes[1,1].set_title('Revolving Balance Distribution')
    
    plt.tight_layout()
    plt.savefig('visualizations/patterns/transaction_patterns.png')
    plt.close()

# 2. Customer Engagement Visualization
def plot_engagement_patterns(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Inactive Months Distribution
    sns.histplot(data=df, x='Months_Inactive_12_mon', hue='Attrition_Flag',
                multiple="dodge", shrink=.8, ax=axes[0])
    axes[0].set_title('Inactive Months Distribution')
    
    # Relationship Count
    sns.boxplot(x='Attrition_Flag', y='Total_Relationship_Count', data=df, ax=axes[1])
    axes[1].set_title('Relationship Count by Churn Status')
    
    plt.tight_layout()
    plt.savefig('visualizations/patterns/engagement_patterns.png')
    plt.close()

# 3. Feature Importance Visualization
def plot_feature_importance(importance_dict):
    plt.figure(figsize=(12, 6))
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())
    
    sns.barplot(x=importance, y=features)
    plt.title('Feature Importance in Churn Prediction')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('visualizations/features/feature_importance.png')
    plt.close()

# 4. Preprocessing Impact Visualization
def plot_preprocessing_impact(df_before, df_after, feature):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Before transformation
    sns.histplot(data=df_before, x=feature, ax=axes[0])
    axes[0].set_title(f'{feature} Before Transformation')
    
    # After transformation
    sns.histplot(data=df_after, x=feature, ax=axes[1])
    axes[1].set_title(f'{feature} After Transformation')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/preprocessing/transform_impact_{feature}.png')
    plt.close()

# 5. Derived Features Analysis
def plot_derived_features(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    derived_features = [
        'Trans_to_Credit_Ratio',
        'Avg_Transaction_per_Contact',
        'Inactive_Contact_Ratio',
        'Relationship_Tenure'
    ]
    
    for i, feature in enumerate(derived_features):
        row = i // 2
        col = i % 2
        sns.boxplot(x='Attrition_Flag', y=feature, data=df, ax=axes[row,col])
        axes[row,col].set_title(f'{feature} by Churn Status')
    
    plt.tight_layout()
    plt.savefig('visualizations/features/derived_features.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('data/BankChurners.csv')
    
    # Create all visualizations
    plot_transaction_patterns(df)
    plot_engagement_patterns(df)
    
    # Example feature importance dict
    importance_dict = {
        'Total_Trans_Amt': 21.5,
        'Total_Trans_Ct': 21.2,
        'Total_Revolving_Bal': 12.0,
        'Avg_Utilization_Ratio': 7.9,
        'Total_Relationship_Count': 7.5
    }
    plot_feature_importance(importance_dict)
    
    # Store original data for preprocessing comparison
    df_before = df.copy()
    # Apply transformations (example for Total_Trans_Amt)
    df_after = df.copy()
    df_after['Total_Trans_Amt'] = np.log1p(df_after['Total_Trans_Amt'])
    plot_preprocessing_impact(df_before, df_after, 'Total_Trans_Amt')
    
    # Create derived features and plot
    df['Trans_to_Credit_Ratio'] = df['Total_Trans_Amt'] / df['Credit_Limit']
    df['Avg_Transaction_per_Contact'] = df['Total_Trans_Amt'] / df['Contacts_Count_12_mon'].replace(0, 1)
    df['Inactive_Contact_Ratio'] = df['Months_Inactive_12_mon'] / df['Contacts_Count_12_mon'].replace(0, 1)
    df['Relationship_Tenure'] = df['Months_on_book'] / df['Total_Relationship_Count']
    plot_derived_features(df) 