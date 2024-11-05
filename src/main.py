import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import mstats
import joblib 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from pathlib import Path

# Define paths
MODEL_DIR = Path("models")
SCALER_DIR = MODEL_DIR / "scalers"
DATA_DIR = Path("data")

# Create necessary directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)
Path("visualizations").mkdir(exist_ok=True)

print("1. Loading and preprocessing data...")
df = pd.read_csv(DATA_DIR / 'BankChurners.csv')

# Remove Naive Bayes columns and handle unknown values
columns_to_drop = [col for col in df.columns if 'Naive_Bayes_Classifier' in col]
df = df.drop(columns=columns_to_drop)
categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category']
df = df[~df[categorical_cols].isin(['Unknown']).any(axis=1)]

# Add correlation heatmap code here
print("Creating correlation heatmap...")
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])
# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

# Create a larger figure for better readability
plt.figure(figsize=(12, 10))
# Create heatmap with improved styling
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Use a diverging colormap
            center=0,  # Center the colormap at 0
            square=True,  # Make the plot square-shaped
            fmt='.2f',  # Round correlation values to 2 decimal places
            linewidths=0.5,  # Add gridlines
            cbar_kws={"shrink": .8})  # Adjust colorbar size

plt.title('Feature Correlation Heatmap', pad=20, size=16)
plt.tight_layout()
# Save the plot
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("2. Creating derived features...")
# Create derived features
df['Trans_to_Credit_Ratio'] = df['Total_Trans_Amt'] / df['Credit_Limit']
df['Avg_Transaction_per_Contact'] = df['Total_Trans_Amt'] / df['Contacts_Count_12_mon'].replace(0, 1)
df['Inactive_Contact_Ratio'] = df['Months_Inactive_12_mon'] / df['Contacts_Count_12_mon'].replace(0, 1)
df['Relationship_Tenure'] = df['Months_on_book'] / df['Total_Relationship_Count']

print("3. Encoding and transforming data...")
# Encode categorical variables
le_education = LabelEncoder()
le_income = LabelEncoder()

# Define the order for education and income
education_order = ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate']
income_order = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']

# Clean up the income categories first
df['Income_Category'] = df['Income_Category'].str.strip()  # Remove any whitespace
df['Income_Category'] = pd.Categorical(df['Income_Category'], 
                                     categories=income_order, 
                                     ordered=True)

# Clean up education levels
df['Education_Level'] = df['Education_Level'].str.strip()  # Remove any whitespace
df['Education_Level'] = pd.Categorical(df['Education_Level'], 
                                     categories=education_order, 
                                     ordered=True)

# Apply label encoding
df['Education_Level_Encoded'] = le_education.fit_transform(df['Education_Level'])
df['Income_Category_Encoded'] = le_income.fit_transform(df['Income_Category'])

# One-Hot Encoding for nominal variables
df_encoded = pd.get_dummies(df, columns=['Gender', 'Marital_Status', 'Card_Category'])

# Handle outliers first
df_encoded['Total_Trans_Amt'] = mstats.winsorize(df_encoded['Total_Trans_Amt'], limits=[0.01, 0.01])
df_encoded['Credit_Limit'] = mstats.winsorize(df_encoded['Credit_Limit'], limits=[0.01, 0.01])

# Log transformation
epsilon = 1e-10  # Small constant to avoid log(0)
df_encoded['Total_Trans_Amt'] = np.log1p(df_encoded['Total_Trans_Amt'] + epsilon)
df_encoded['Credit_Limit'] = np.log1p(df_encoded['Credit_Limit'] + epsilon)

# Standard scaling
std_scaler = StandardScaler()
std_columns = ['Credit_Limit', 'Total_Trans_Amt']
df_encoded[std_columns] = std_scaler.fit_transform(df_encoded[std_columns])

# Handle NaN values separately for numerical and categorical columns
numerical_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df_encoded.select_dtypes(include=['category']).columns

# Fill NaN values in numerical columns with mean
if len(numerical_columns) > 0:
    df_encoded[numerical_columns] = df_encoded[numerical_columns].fillna(df_encoded[numerical_columns].mean())

# Fill NaN values in categorical columns with mode
if len(categorical_columns) > 0:
    df_encoded[categorical_columns] = df_encoded[categorical_columns].fillna(df_encoded[categorical_columns].mode().iloc[0])

# MinMax scaling for other numerical features
minmax_scaler = MinMaxScaler()
minmax_columns = ['Customer_Age', 'Months_on_book', 'Total_Relationship_Count', 
                 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Total_Trans_Ct', 
                 'Avg_Utilization_Ratio', 'Trans_to_Credit_Ratio', 
                 'Avg_Transaction_per_Contact', 'Inactive_Contact_Ratio', 
                 'Relationship_Tenure']
df_encoded[minmax_columns] = minmax_scaler.fit_transform(df_encoded[minmax_columns])

# Prepare target variable and final dataset
df_encoded['Attrition_Flag'] = (df_encoded['Attrition_Flag'] == 'Attrited Customer').astype(int)
columns_to_remove = ['CLIENTNUM', 'Education_Level', 'Income_Category', 'Avg_Open_To_Buy']
final_df = df_encoded.drop(columns=columns_to_remove)

# Save feature names
feature_names = final_df.drop('Attrition_Flag', axis=1).columns.tolist()
pd.Series(feature_names).to_csv('feature_names.csv', index=False)

# Create train-test-validation split
print("4. Creating data splits...")
X = final_df.drop('Attrition_Flag', axis=1)
y = final_df['Attrition_Flag']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# Save the processed data
print("5. Saving processed data...")
np.save(MODEL_DIR / 'X_train.npy', X_train)
np.save(MODEL_DIR / 'X_val.npy', X_val)
np.save(MODEL_DIR / 'X_test.npy', X_test)
np.save(MODEL_DIR / 'y_train.npy', y_train)
np.save(MODEL_DIR / 'y_val.npy', y_val)
np.save(MODEL_DIR / 'y_test.npy', y_test)

# Save transformers
joblib.dump(minmax_scaler, SCALER_DIR / 'minmax_scaler.joblib')
joblib.dump(std_scaler, SCALER_DIR / 'std_scaler.joblib')

encoders = {
    'education': le_education,
    'income': le_income
}
joblib.dump(encoders, SCALER_DIR / 'label_encoders.joblib')

print("Data preprocessing completed successfully!")