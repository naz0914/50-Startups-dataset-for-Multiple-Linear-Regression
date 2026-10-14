"""
AIoT-DA L6 Multiple Linear Regression (Kaggle Boston Housing) with ChatGPT
CRISP-DM Full End-to-End Workflow Implementation

Course Focus:
1. CRISP-DM Standard (Business Understanding -> Data Understanding -> Data Prep -> Modeling -> Evaluation -> Deployment)
2. Train/Test Split (80/20) & Feature Scaling
3. Evaluation Metrics: MSE, MAE, RMSE, R2 Interpretation
4. Overfitting & Underfitting Diagnosis (Learning Curves)
5. Feature Selection (Forward Selection, Lasso, RFE, Exhaustive Best Subset)
6. Model Deployment & Serialization (pickle, gzip)
"""

import os
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ==========================================
# CRISP-DM Step 1: Business Understanding
# ==========================================
print_header("CRISP-DM Step 1: Business Understanding (Boston Housing)")
print("""
[Business Objective]
Develop a predictive valuation model to estimate median housing prices (MEDV, in $1,000s)
in the Boston metropolitan area based on 13 socioeconomic, environmental, and structural factors:
- RM (Rooms per dwelling)
- LSTAT (% Lower status population)
- PTRATIO (Pupil-teacher ratio)
- DIS (Distance to employment centres)
- CRIM (Per capita crime rate)
- TAX (Property-tax rate)
- CHAS (Charles River dummy variable)
- NOX, INDUS, ZN, AGE, RAD, B

[Stakeholders]
Real Estate Investors, Urban Planners, Property Tax Assessors, Homebuyers.

[Success Criteria]
Achieve test R2 > 0.65, low RMSE, and stable generalization without overfitting.
""")


# ==========================================
# CRISP-DM Step 2: Data Understanding
# ==========================================
print_header("CRISP-DM Step 2: Data Understanding")

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
print(f"Loading Boston Housing dataset from: {url}")
df = pd.read_csv(url)
df.columns = [c.upper() for c in df.columns]

print(f"\nDataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values Count:")
print(df.isnull().sum())

print("\nDescriptive Statistics (Key Features):")
print(df[['RM', 'LSTAT', 'PTRATIO', 'CRIM', 'DIS', 'MEDV']].describe().round(2))

# Data Exploration: Scatter Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
key_feats = ['RM', 'LSTAT', 'PTRATIO']
colors = ['#2563eb', '#ef4444', '#10b981']

for ax, feat, color in zip(axes, key_feats, colors):
    sns.regplot(x=feat, y='MEDV', data=df, ax=ax, color=color, scatter_kws={'alpha': 0.6})
    ax.set_title(f'{feat} vs MEDV (House Price)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MEDV ($1,000s)')
    ax.set_xlabel(feat)

plt.tight_layout()
plt.savefig('boston_scatter_exploration.png', dpi=300)
plt.close()
print("Saved exploratory scatter plot as 'boston_scatter_exploration.png'")


# ==========================================
# CRISP-DM Step 3: Data Preparation
# ==========================================
print_header("CRISP-DM Step 3: Data Preparation")

features = [c for c in df.columns if c != 'MEDV']
X = df[features]
y = df['MEDV']

print(f"Features ({len(features)}): {features}")
print(f"Target: MEDV (Median home value in $1,000s)")

# Train/Test Split (80/20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples (80%)")
print(f"Testing set size:  {X_test.shape[0]} samples (20%)")


# ==========================================
# CRISP-DM Step 4: Modeling
# ==========================================
print_header("CRISP-DM Step 4: Modeling")

lr_full = LinearRegression()
lr_full.fit(X_train, y_train)

print(f"Model Intercept (b0): {lr_full.intercept_:.4f}")
print("Model Coefficients:")
for feat, coef in zip(features, lr_full.coef_):
    print(f"  - {feat:10s}: {coef:10.4f}")


# ==========================================
# CRISP-DM Step 5: Evaluation & Diagnosis
# ==========================================
print_header("CRISP-DM Step 5: Evaluation & Diagnosis")

y_pred_full = lr_full.predict(X_test)

r2 = r2_score(y_test, y_pred_full)
mae = mean_absolute_error(y_test, y_pred_full)
mse = mean_squared_error(y_test, y_pred_full)
rmse = np.sqrt(mse)

print(f"R-squared (R2 Score):        {r2:.6f}  (Explains {r2*100:.2f}% of test price variance)")
print(f"Mean Absolute Error (MAE):   ${mae*1000:,.2f}")
print(f"Mean Squared Error (MSE):    {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse*1000:,.2f}")

# Overfit/Underfit Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    LinearRegression(), X, y,
    cv=5,
    train_sizes=np.linspace(0.2, 1.0, 8),
    scoring='r2',
    random_state=42
)

plt.figure(figsize=(8, 5))
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
plt.plot(train_sizes, train_mean, 'o-', color='#2563eb', label='Training R2')
plt.plot(train_sizes, val_mean, 's-', color='#10b981', label='Cross-Validation R2')
plt.title('Boston Housing: Learning Curve Diagnosis', fontsize=12, fontweight='bold')
plt.xlabel('Training Samples')
plt.ylabel('R2 Score')
plt.legend()
plt.tight_layout()
plt.savefig('boston_learning_curve.png', dpi=300)
plt.close()
print("Saved learning curve plot as 'boston_learning_curve.png'")


# ==========================================
# Step 5.5: Feature Selection (Forward Selection)
# ==========================================
print_header("Step 5.5: Forward Feature Selection Benchmark")


def evaluate_subset(feats):
    m = LinearRegression().fit(X_train[feats], y_train)
    p = m.predict(X_test[feats])
    return np.sqrt(mean_squared_error(y_test, p)), r2_score(y_test, p)


selected_sfs = []
results_table = []
for k in range(1, 7):
    pool = [f for f in features if f not in selected_sfs]
    best_f, best_rmse, best_r2 = None, np.inf, -np.inf
    for f in pool:
        curr_rmse, curr_r2 = evaluate_subset(selected_sfs + [f])
        if curr_rmse < best_rmse:
            best_rmse, best_r2, best_f = curr_rmse, curr_r2, f
    selected_sfs.append(best_f)
    results_table.append({
        'Number of Features': k,
        'Selected Features': selected_sfs.copy(),
        'RMSE ($1k)': round(best_rmse, 3),
        'R-squared': round(best_r2, 6)
    })

df_fs = pd.DataFrame(results_table)
print("Forward Selection Results Table (Top 1..6):")
print(df_fs.to_string(index=False))


# ==========================================
# CRISP-DM Step 6: Deployment
# ==========================================
print_header("CRISP-DM Step 6: Deployment & Inference")

# Optimal Pruned Model (Top 4 Features: LSTAT, RM, PTRATIO, DIS)
top_feats = selected_sfs[:4]
optimal_model = LinearRegression().fit(X_train[top_feats], y_train)

# Save models
with open('boston_model.pkl', 'wb') as f:
    pickle.dump({'model': optimal_model, 'features': top_feats}, f)
with gzip.open('boston_model.pkl.gz', 'wb') as f:
    pickle.dump({'model': optimal_model, 'features': top_feats}, f)
print("1. Serialized optimal Boston Housing model to 'boston_model.pkl' & 'boston_model.pkl.gz'")

# Production Inference Demonstration
sample_house = pd.DataFrame([[6.5, 9.8, 17.4, 4.2]], columns=top_feats)
predicted_price = optimal_model.predict(sample_house)[0]

print("\n--- Production Valuation Demonstration ---")
print(f"Input: Rooms (RM)=6.5, Low-Status % (LSTAT)=9.8%, Pupil-Teacher (PTRATIO)=17.4, Distance (DIS)=4.2km")
print(f"Predicted Median Home Price: ${predicted_price * 1000:,.2f}")
print("============================================================")
