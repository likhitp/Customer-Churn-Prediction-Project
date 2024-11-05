import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Create visualizations directory if it doesn't exist
if not os.path.exists('model_visualizations'):
    os.makedirs('model_visualizations')

# Update these paths
MODEL_DIR = Path("models")
SCALER_DIR = MODEL_DIR / "scalers"
DATA_DIR = Path("data")

def load_data():
    """Load preprocessed data and feature names."""
    try:
        X_train = np.load(MODEL_DIR / 'X_train.npy', allow_pickle=True)
        X_val = np.load(MODEL_DIR / 'X_val.npy', allow_pickle=True)
        X_test = np.load(MODEL_DIR / 'X_test.npy', allow_pickle=True)
        y_train = np.load(MODEL_DIR / 'y_train.npy', allow_pickle=True)
        y_val = np.load(MODEL_DIR / 'y_val.npy', allow_pickle=True)
        y_test = np.load(MODEL_DIR / 'y_test.npy', allow_pickle=True)
        feature_names = pd.read_csv('feature_names.csv')['0'].tolist()
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def plot_roc_curves(models, X_val, y_val):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        auc = roc_auc_score(y_val, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend()
    plt.savefig('model_visualizations/roc_curves.png')
    plt.close()

def validate_and_prepare_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Validate and prepare data for modeling."""
    # Convert pandas DataFrames to numpy arrays if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    
    # Convert to float32 for better compatibility
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Handle any remaining NaN values
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)
    
    print("Data shapes:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_smote(X_train, y_train):
    """Apply SMOTE to balance the training data."""
    print("\nApplying SMOTE...")
    print("Original class distribution:", np.bincount(y_train))
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print("Class distribution after SMOTE:", np.bincount(y_train_smote))
    return X_train_smote, y_train_smote

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for the model."""
    plt.figure(figsize=(12, 6))
    
    if model_name == "Logistic Regression":
        importance = abs(model.coef_[0])
    elif model_name in ["Random Forest", "XGBoost"]:
        importance = model.feature_importances_
    
    # Create DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot top 15 features
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title(f'Top 15 Important Features - {model_name}')
    plt.tight_layout()
    plt.savefig(f'model_visualizations/feature_importance_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_precision_recall_curve(models, X_val, y_val):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
        avg_precision = average_precision_score(y_val, y_pred_proba)
        plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for All Models')
    plt.legend()
    plt.savefig('model_visualizations/precision_recall_curves.png')
    plt.close()

def main():
    # Load data
    print("1. Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data()

    # Validate and prepare data
    print("2. Validating and preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = validate_and_prepare_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # Apply SMOTE
    print("3. Applying SMOTE...")
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Define models with improved parameters
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,  # Increased from 1000
            random_state=42,
            class_weight='balanced',  # Added class weighting
            solver='lbfgs',
            n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,  # Increased from 100
            max_depth=10,      # Added depth limit
            min_samples_split=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
    }

    # Train and evaluate models
    model_performances = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_smote, y_train_smote)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        # Store performance metrics
        model_performances[name] = {
            'classification_report': classification_report(y_val, y_val_pred),
            'confusion_matrix': confusion_matrix(y_val, y_val_pred),
            'roc_auc': roc_auc_score(y_val, y_val_proba),
            'avg_precision': average_precision_score(y_val, y_val_proba)
        }
        
        # Print performance metrics
        print(f"\n{name} Validation Results:")
        print("Classification Report:")
        print(model_performances[name]['classification_report'])
        print("\nConfusion Matrix:")
        print(model_performances[name]['confusion_matrix'])
        print(f"\nROC AUC Score: {model_performances[name]['roc_auc']:.4f}")
        print(f"Average Precision Score: {model_performances[name]['avg_precision']:.4f}")
        
        # Plot feature importance
        plot_feature_importance(model, feature_names, name)

    # Plot ROC and Precision-Recall curves
    plot_roc_curves(models, X_val, y_val)
    plot_precision_recall_curve(models, X_val, y_val)

    # Select best model based on both ROC AUC and Average Precision
    best_model_name = max(model_performances.items(), 
                         key=lambda x: (x[1]['roc_auc'] + x[1]['avg_precision'])/2)[0]
    print(f"\nBest performing model: {best_model_name}")

    # Save best model and its feature importance
    joblib.dump(models[best_model_name], MODEL_DIR / 'best_model.joblib')
    print(f"Best model saved in models directory")

    # Save performance metrics
    with open(MODEL_DIR / 'model_performances.txt', 'w') as f:
        for name, metrics in model_performances.items():
            f.write(f"\n{name} Performance Metrics:\n")
            f.write(f"ROC AUC Score: {metrics['roc_auc']:.4f}\n")
            f.write(f"Average Precision Score: {metrics['avg_precision']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(metrics['classification_report'])
            f.write("\nConfusion Matrix:\n")
            f.write(str(metrics['confusion_matrix']))
            f.write("\n" + "="*50 + "\n")

    # Instead of separate files for each encoder
    encoders = {
        'Education': LabelEncoder(),
        'Income': LabelEncoder(),
        # ... other encoders
    }

    # Save all encoders in one file
    joblib.dump(encoders, 'models/scalers/label_encoders.joblib')

if __name__ == "__main__":
    main()