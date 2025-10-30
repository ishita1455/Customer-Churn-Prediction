# =====================================================
# Train Customer Churn Model
# =====================================================

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 Customer Churn Model Training Pipeline (Anti-Overfitting)")
print("=" * 60)

# =====================================================
# 1. Load Dataset
# =====================================================
print("\n📂 Step 1: Loading Dataset...")
try:
    df = pd.read_csv("customer_dataset.csv")
    print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
except FileNotFoundError:
    print("❌ Error: 'customer_dataset.csv' not found!")
    print("   Please place the dataset in the same directory as this script.")
    exit()

# =====================================================
# 2. Data Cleaning
# =====================================================
print("\n🧹 Step 2: Data Cleaning...")

# Fill missing values
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].fillna(df['Churn'].mode()[0])

# Drop duplicates
initial_rows = len(df)
df.drop_duplicates(inplace=True)
print(f"   Removed {initial_rows - len(df)} duplicate rows")

# Drop unnecessary columns
drop_cols = ["Customer ID", "Customer Name"]
for col in drop_cols:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"   Dropped: {col}")

print("✅ Data cleaning completed")

# =====================================================
# 3. Feature Engineering
# =====================================================
print("\n⚙️ Step 3: Feature Engineering...")

# Parse Purchase Date if present
if "Purchase Date" in df.columns:
    try:
        df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="coerce")
        df["purchase_year"] = df["Purchase Date"].dt.year.fillna(0).astype(int)
        df["purchase_month"] = df["Purchase Date"].dt.month.fillna(0).astype(int)
        df["purchase_day"] = df["Purchase Date"].dt.day.fillna(0).astype(int)
        df.drop(columns=["Purchase Date"], inplace=True)
        print("   Created: purchase_year, purchase_month, purchase_day")
    except Exception as e:
        print(f"   Warning: Could not parse Purchase Date - {e}")

# Map Returns to binary
if "Returns" in df.columns:
    df["Returns"] = df["Returns"].astype(str).str.lower().map(
        lambda x: 1 if x in ["yes", "1", "true", "returned"] else 0
    )

print("✅ Feature engineering completed")

# =====================================================
# 4. Label Encoding
# =====================================================
print("\n🔤 Step 4: Encoding Categorical Features...")

label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns.tolist()

for col in categorical_cols:
    if col != 'Churn':  # Don't encode target yet
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded: {col} ({len(le.classes_)} classes)")

print("✅ Encoding completed")

# =====================================================
# 5. Feature Scaling
# =====================================================
print("\n📊 Step 5: Scaling Numeric Features...")

# Identify numeric columns (excluding Churn)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Churn' in numeric_cols:
    numeric_cols.remove('Churn')

# Scale numeric features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(f"   Scaled {len(numeric_cols)} numeric columns")

print("✅ Scaling completed")

# =====================================================
# 6. Train-Test Split
# =====================================================
print("\n✂️ Step 6: Train-Test Split...")

assert "Churn" in df.columns, "❌ Error: 'Churn' column not found in dataset!"

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")
print(f"   Class distribution (train): Class 0={sum(y_train==0)}, Class 1={sum(y_train==1)}")
print(f"   Class distribution (test): Class 0={sum(y_test==0)}, Class 1={sum(y_test==1)}")

# =====================================================
# 7. SMOTE Balancing (Conservative)
# =====================================================
print("\n⚖️ Step 7: Applying Conservative SMOTE...")

# Use conservative sampling to prevent overfitting
minority_class_count = sum(y_train == 1)
majority_class_count = sum(y_train == 0)

# Only balance to 70% of majority class to avoid overfitting
target_minority = int(majority_class_count * 0.7)

smote = SMOTE(
    sampling_strategy={1: target_minority},
    random_state=42,
    k_neighbors=3  # Reduced from default 5 to prevent overfitting
)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"   After SMOTE:")
print(f"   Class 0: {sum(y_train_res==0)} samples")
print(f"   Class 1: {sum(y_train_res==1)} samples")
print(f"   Balance ratio: {sum(y_train_res==1)/sum(y_train_res==0):.2f}")

print("✅ SMOTE applied successfully")

# =====================================================
# 8. Optuna Hyperparameter Tuning (Anti-Overfitting)
# =====================================================
print("\n🔍 Step 8: Hyperparameter Tuning with Optuna...")
print("   Focus: Preventing overfitting while maintaining recall...")

def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150, step=25),
        'max_depth': trial.suggest_int('max_depth', 6, 15),  # Reduced depth
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 30),  # Increased
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 15),  # Increased
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'max_samples': trial.suggest_float('max_samples', 0.6, 0.9),  # Bootstrap sampling
        'class_weight': 'balanced_subsample',  # Better for imbalanced data
        'random_state': 42,
        'n_jobs': -1
    }
    
    rf = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Focus on F1 score to balance precision and recall
    f1_scores = cross_val_score(rf, X_train_res, y_train_res, cv=cv, scoring='f1', n_jobs=-1)
    recall_scores = cross_val_score(rf, X_train_res, y_train_res, cv=cv, scoring='recall', n_jobs=-1)
    
    # Penalize models with high variance (sign of overfitting)
    f1_mean = f1_scores.mean()
    f1_std = f1_scores.std()
    recall_mean = recall_scores.mean()
    
    # Combined score: prioritize recall for churn detection, penalize variance
    combined_score = (0.4 * f1_mean) + (0.5 * recall_mean) - (0.1 * f1_std)
    
    return combined_score

# Run Optuna study
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(rf_objective, n_trials=10, timeout=600, show_progress_bar=True)

print(f"\n🏆 Best trial: #{study.best_trial.number}")
print(f"   Best Combined Score: {study.best_value:.4f}")
print(f"   Best Parameters:")
for key, value in study.best_params.items():
    print(f"      {key}: {value}")

# =====================================================
# 9. Train Final Model
# =====================================================
print("\n🤖 Step 9: Training Final Model with Best Parameters...")

best_params = study.best_params
best_model = RandomForestClassifier(
    **best_params,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train_res, y_train_res)
print("✅ Model training completed")

# =====================================================
# 10. Model Evaluation (Train vs Test)
# =====================================================
print("\n📈 Step 10: Model Evaluation (Checking for Overfitting)...")

# Train set evaluation
y_train_pred = best_model.predict(X_train_res)
train_accuracy = accuracy_score(y_train_res, y_train_pred)
train_f1 = f1_score(y_train_res, y_train_pred)
train_recall = recall_score(y_train_res, y_train_pred)

# Test set evaluation
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

print("\n" + "=" * 60)
print("📊 MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print("\n🔵 TRAIN SET PERFORMANCE:")
print(f"   Accuracy : {train_accuracy:.4f}")
print(f"   F1 Score : {train_f1:.4f}")
print(f"   Recall   : {train_recall:.4f}")

print("\n🟢 TEST SET PERFORMANCE:")
print(f"   Accuracy : {test_accuracy:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall   : {test_recall:.4f} ⭐ (High-risk detection)")
print(f"   F1 Score : {test_f1:.4f}")
print(f"   ROC-AUC  : {test_auc:.4f}")

# Overfitting check
print("\n🔍 OVERFITTING CHECK:")
accuracy_gap = train_accuracy - test_accuracy
f1_gap = train_f1 - test_f1
print(f"   Accuracy gap (train-test): {accuracy_gap:.4f}")
print(f"   F1 Score gap (train-test): {f1_gap:.4f}")

if accuracy_gap < 0.05 and f1_gap < 0.05:
    print("   ✅ Model shows LOW overfitting - good generalization!")
elif accuracy_gap < 0.10 and f1_gap < 0.10:
    print("   ⚠️  Model shows MODERATE overfitting - acceptable")
else:
    print("   ❌ Model shows HIGH overfitting - consider further regularization")

print("\n📋 Confusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\n   True Negatives : {cm[0,0]}")
print(f"   False Positives: {cm[0,1]}")
print(f"   False Negatives: {cm[1,0]} ⚠️  (Missed high-risk customers)")
print(f"   True Positives : {cm[1,1]} ✅ (Correctly identified churn)")

print("\n📄 Classification Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Feature importance
print("\n🎯 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# =====================================================
# 11. Save Models
# =====================================================
print("\n💾 Step 11: Saving Models to 'models_trained' folder...")

# Create models_saved directory
models_dir = Path("models_trained")
models_dir.mkdir(exist_ok=True)

# Save the trained model
model_path = models_dir / "best_churn_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"   ✅ Model saved: {model_path.resolve()}")

# Save preprocessors (encoders and scaler)
preprocessor_path = models_dir / "preprocessors.pkl"
preprocessors = {
    "label_encoders": label_encoders,
    "scaler": scaler,
    "feature_names": list(X.columns)
}
with open(preprocessor_path, "wb") as f:
    pickle.dump(preprocessors, f)
print(f"   ✅ Preprocessors saved: {preprocessor_path.resolve()}")

# Save model metadata
metadata_path = models_dir / "model_metadata.txt"
with open(metadata_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("CUSTOMER CHURN MODEL METADATA\n")
    f.write("(Anti-Overfitting Version)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training Date: {pd.Timestamp.now()}\n\n")
    f.write(f"Dataset Shape: {df.shape}\n")
    f.write(f"Features: {list(X.columns)}\n\n")
    
    f.write("Train Set Performance:\n")
    f.write(f"  Accuracy : {train_accuracy:.4f}\n")
    f.write(f"  F1 Score : {train_f1:.4f}\n")
    f.write(f"  Recall   : {train_recall:.4f}\n\n")
    
    f.write("Test Set Performance:\n")
    f.write(f"  Accuracy : {test_accuracy:.4f}\n")
    f.write(f"  Precision: {test_precision:.4f}\n")
    f.write(f"  Recall   : {test_recall:.4f}\n")
    f.write(f"  F1 Score : {test_f1:.4f}\n")
    f.write(f"  ROC-AUC  : {test_auc:.4f}\n\n")
    
    f.write("Overfitting Metrics:\n")
    f.write(f"  Accuracy gap: {accuracy_gap:.4f}\n")
    f.write(f"  F1 Score gap: {f1_gap:.4f}\n\n")
    
    f.write("Best Hyperparameters:\n")
    for key, value in best_params.items():
        f.write(f"  {key}: {value}\n")
    
    f.write("\nTop 10 Important Features:\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

print(f"   ✅ Metadata saved: {metadata_path.resolve()}")

print("\n" + "=" * 60)
print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📁 All files saved in: {models_dir.resolve()}")
print("   - best_churn_model.pkl")
print("   - preprocessors.pkl")
print("   - model_metadata.txt")
print("\n✨ Model optimized for:")
print("   • Low overfitting (good generalization)")
print("   • High recall (catches high-risk churn customers)")
print("   • Balanced precision-recall trade-off")
print("\n🚀 Ready to use with Streamlit app!")