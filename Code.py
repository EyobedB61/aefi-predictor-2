import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import os
import warnings
import logging
import joblib

# Configure logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define data directory
data_dir = r"C:\Users\hp\Desktop\Thesis\data set\10 Year Data Set"
logging.info(f"Data directory set to: {data_dir}")

# 1. Data Loading and Combining
def load_and_combine_data():
    dfs = []
    core_columns = ['ID', 'AGE_YRS', 'SEX', 'VAX_TYPE', 'VAX_DOSE_SERIES', 'VAX_NAME', 
                    'VAERS_ID', 'DIED', 'HOSPITAL', 'VAX_DATE', 'ONSET_DATE', 'NUMDAYS',
                    'SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4', 'SYMPTOM5']
    logging.info("Core columns defined: %s", core_columns)
    
    for year in range(2016, 2026):
        file_path = os.path.join(data_dir, f"{year} data.csv")
        logging.info(f"Checking file: {file_path}")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, low_memory=False)
                logging.info(f"Loaded {file_path} with shape: {df.shape}")
                available_columns = [col for col in core_columns if col in df.columns]
                logging.info(f"Available columns in {file_path}: {available_columns}")
                if not available_columns:
                    logging.warning(f"No core columns found in {file_path}. Skipping.")
                    continue
                df = df[available_columns]
                dfs.append(df)
                logging.info(f"Appended {file_path} with {len(df)} records.")
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
        else:
            logging.warning(f"File {file_path} not found.")
    
    if not dfs:
        logging.error("No data files loaded. Check data directory and file names.")
        raise ValueError("No data files loaded.")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.columns = combined_df.columns.str.upper().str.strip()
    logging.info(f"Combined dataset: {len(combined_df)} records, {len(combined_df.columns)} columns.")
    logging.info(f"Columns in combined dataset: {list(combined_df.columns)}")
    
    return combined_df

# Load data
logging.info("Starting data loading...")
try:
    df = load_and_combine_data()
except Exception as e:
    logging.error(f"Failed to load data: {e}")
    raise

# 2. Data Cleaning and Preprocessing
logging.info("Starting data cleaning...")

# Log null counts
logging.info("Null counts before preprocessing:")
logging.info(df.isnull().sum().to_string())

# Define AEFI: Exclude non-adverse symptoms like "No adverse event"
def is_adverse_symptom(symptom):
    if pd.isna(symptom):
        return False
    non_adverse = ['NO ADVERSE EVENT', 'NONE']
    return symptom.strip().upper() not in non_adverse

if all(col in df.columns for col in ['SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4', 'SYMPTOM5']):
    df['AEFI'] = (
        df[['SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4', 'SYMPTOM5']].applymap(is_adverse_symptom).any(axis=1) |
        df[['DIED', 'HOSPITAL']].eq('Y').any(axis=1, skipna=True)
    ).astype(int)
else:
    df['AEFI'] = df[['DIED', 'HOSPITAL']].eq('Y').any(axis=1, skipna=True).astype(int)
    logging.warning("Some symptom columns missing. AEFI based on DIED and HOSPITAL only.")

logging.info("AEFI distribution:")
logging.info(df['AEFI'].value_counts().to_string())

# Check if AEFI has both classes
if len(df['AEFI'].unique()) < 2:
    logging.error("AEFI contains only one class: %s. Cannot proceed with modeling.", df['AEFI'].unique())
    raise ValueError("AEFI contains only one class. Inspect data for non-AEFI cases.")

# Handle missing values
logging.info("Handling missing values...")

# Numerical: AGE_YRS, NUMDAYS
numerical_cols = [col for col in ['AGE_YRS', 'NUMDAYS'] if col in df.columns]
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    logging.info(f"Converted {col} to numeric. Null count: {df[col].isnull().sum()}")

if numerical_cols:
    imputer = KNNImputer(n_neighbors=5)
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols].fillna(df[numerical_cols].median()))
    logging.info(f"Imputed numerical columns: {numerical_cols}")

# Categorical: SEX, VAX_TYPE, VAX_DOSE_SERIES
categorical_cols = [col for col in ['SEX', 'VAX_TYPE', 'VAX_DOSE_SERIES'] if col in df.columns]
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    logging.info(f"Imputed {col} with mode. Unique values: {df[col].nunique()}")

# Convert dates and calculate time since vaccination
if 'VAX_DATE' in df.columns and 'ONSET_DATE' in df.columns:
    df['VAX_DATE'] = pd.to_datetime(df['VAX_DATE'], errors='coerce')
    df['ONSET_DATE'] = pd.to_datetime(df['ONSET_DATE'], errors='coerce')
    df['TIME_SINCE_VAX'] = (df['ONSET_DATE'] - df['VAX_DATE']).dt.days
    df['TIME_SINCE_VAX'] = df['TIME_SINCE_VAX'].fillna(df['NUMDAYS'] if 'NUMDAYS' in df.columns else df['TIME_SINCE_VAX'].median())
    logging.info(f"TIME_SINCE_VAX created. Null count: {df['TIME_SINCE_VAX'].isnull().sum()}")
else:
    df['TIME_SINCE_VAX'] = df['NUMDAYS'] if 'NUMDAYS' in df.columns else 0
    logging.warning("VAX_DATE or ONSET_DATE missing. TIME_SINCE_VAX set to NUMDAYS or 0.")

# Outlier detection and treatment
def cap_outliers(series, col_name):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    logging.info(f"{col_name} - Q1: {Q1}, Q3: {Q3}, IQR: {IQR}, Lower: {lower_bound}, Upper: {upper_bound}")
    return series.clip(lower_bound, upper_bound)

for col in ['AGE_YRS', 'TIME_SINCE_VAX']:
    if col in df.columns:
        df[col] = cap_outliers(df[col], col)
        logging.info(f"Capped outliers for {col}")

# 3. Exploratory Data Analysis (EDA)
logging.info("Starting EDA...")

# Age and Sex Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='AGE_YRS', hue='SEX', bins=30, kde=True, multiple='stack')
plt.title('Age and Sex Distribution')
plt.xlabel('Age (Years)')
plt.ylabel('Count')
plt.savefig('age_sex_distribution.png')
plt.close()
logging.info("Saved age_sex_distribution.png")

# AEFI by Vaccine type
if 'VAX_TYPE' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='VAX_TYPE', hue='AEFI')
    plt.title('AEFI by Vaccine Type')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.savefig('aefi_by_vax_type.png')
    plt.close()
    logging.info("Saved aefi_by_vax_type.png")
else:
    logging.warning("VAX_TYPE column not found. Skipping AEFI by Vaccine Type plot.")

# Feature Importance - XGBoost
try:
    xgb_model = models['XGBoost']
    
    # Create DataFrame with feature importances
    xgb_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Plot all features
    plt.figure(figsize=(12, max(6, len(xgb_importance) * 0.3)))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=xgb_importance,
        palette='plasma'
    )
    plt.title('Feature Importance (XGBoost) - All Features')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    plt.close()
    logging.info("Saved xgboost_feature_importance.png")

    # Save as CSV
    xgb_importance.to_csv('xgboost_feature_importance.csv', index=False)
    logging.info("Saved xgboost_feature_importance.csv")

    # Quick print of top 10
    print("Top 10 Features (XGBoost):")
    print(xgb_importance.head(10))
    
except Exception as e:
    logging.error(f"Error generating XGBoost feature importance plot: {e}")

# Correlation matrix
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=np.number).columns
logging.info(f"Numeric columns for correlation: {numeric_cols}")
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()
logging.info("Saved correlation_matrix.png")

# AEFI distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='AEFI', data=df)
plt.title('AEFI Distribution')
plt.savefig('aefi_distribution.png')
plt.close()
logging.info("Saved aefi_distribution.png")

# 4. Feature Engineering
logging.info("Starting feature engineering...")

# Derived variable: Severe outcome
if 'DIED' in df.columns and 'HOSPITAL' in df.columns:
    df['SEVERE_OUTCOME'] = df[['DIED', 'HOSPITAL']].eq('Y').any(axis=1).astype(int)
    logging.info("SEVERE_OUTCOME created. Distribution: %s", df['SEVERE_OUTCOME'].value_counts().to_string())
else:
    df['SEVERE_OUTCOME'] = 0
    logging.warning("DIED or HOSPITAL missing. SEVERE_OUTCOME set to 0.")

# Interaction term: VAX_TYPE * SEVERE_OUTCOME
if 'VAX_TYPE' in df.columns:
    df['VAX_SEVERE_INTERACTION'] = df['VAX_TYPE'].astype(str) + '_' + df['SEVERE_OUTCOME'].astype(str)
    logging.info("VAX_SEVERE_INTERACTION created. Unique values: %s", df['VAX_SEVERE_INTERACTION'].nunique())
else:
    df['VAX_SEVERE_INTERACTION'] = 'UNKNOWN_' + df['SEVERE_OUTCOME'].astype(str)
    logging.warning("VAX_TYPE missing. VAX_SEVERE_INTERACTION set to UNKNOWN.")

# Encoding categorical variables
ohe_cols = [col for col in ['SEX', 'VAX_TYPE', 'VAX_DOSE_SERIES', 'VAX_SEVERE_INTERACTION'] if col in df.columns]
logging.info(f"Categorical columns for encoding: {ohe_cols}")
if ohe_cols:
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    ohe_data = ohe.fit_transform(df[ohe_cols])
    ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(ohe_cols))
    df = pd.concat([df.drop(ohe_cols, axis=1), ohe_df], axis=1)
    logging.info(f"One-hot encoded columns. New shape: {df.shape}")
else:
    logging.warning("No categorical columns for encoding.")

# Normalization
scaler = StandardScaler()
norm_cols = [col for col in ['AGE_YRS', 'TIME_SINCE_VAX'] if col in df.columns]
if norm_cols:
    df[norm_cols] = scaler.fit_transform(df[norm_cols])
    logging.info(f"Normalized columns: {norm_cols}")
else:
    logging.warning("No columns for normalization.")

# 5. Model Development
logging.info("Starting model development...")

# Select features
exclude_cols = ['AEFI', 'ID', 'VAERS_ID', 'VAX_NAME', 'VAX_DATE', 'ONSET_DATE', 
                'SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4', 'SYMPTOM5', 'DIED', 'HOSPITAL']
X = df.drop([col for col in exclude_cols if col in df.columns], axis=1)
y = df['AEFI']
logging.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")
logging.info(f"Features: {list(X.columns)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
logging.info(f"Train AEFI distribution: %s", pd.Series(y_train).value_counts().to_string())
logging.info(f"Test AEFI distribution: %s", pd.Series(y_test).value_counts().to_string())

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'Neural Network': MLPClassifier(random_state=42, hidden_layer_sizes=(100, 50), max_iter=500)
}
logging.info("Models initialized: %s", list(models.keys()))

# 6. Model Evaluation
results = []
for name, model in models.items():
    logging.info(f"Training {name}...")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_test, y_prob)
        })
        logging.info(f"{name} evaluated successfully. Accuracy: {results[-1]['Accuracy']:.4f}")
    except Exception as e:
        logging.error(f"Error evaluating {name}: {e}")

results_df = pd.DataFrame(results)
logging.info("Model performance:")
logging.info(results_df.to_string())

# Plot ROC curves
plt.figure(figsize=(10, 8))
for name, model in models.items():
    try:
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})')
        logging.info(f"Plotted ROC for {name}")
    except Exception as e:
        logging.error(f"Error plotting ROC for {name}: {e}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('roc_curves.png')
plt.close()
logging.info("Saved roc_curves.png")

# 7. Feature Importance (using Random Forest)
try:
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance (Random Forest)')
    plt.savefig('feature_importance.png')
    plt.close()
    logging.info("Saved feature_importance.png")
except Exception as e:
    logging.error(f"Error generating feature importance: {e}")

# Save results
results_df.to_csv('model_performance.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)
logging.info("Results saved: model_performance.csv, feature_importance.csv")


# Save the trained model
joblib.dump(models['Random Forest'], 'best_model.pkl')
print("✅ Model saved as best_model.pkl")

# Save the fitted StandardScaler
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved as scaler.pkl")

# Save the fitted OneHotEncoder
joblib.dump(ohe, 'encoder.pkl')
print("✅ Encoder saved as encoder.pkl")

