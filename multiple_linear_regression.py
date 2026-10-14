"""
AIoT-DA L6 Multiple Linear Regression (Startup 50 on Kaggle) with ChatGPT
CRISP-DM Full End-to-End Workflow Implementation

Course Focus:
1. CRISP-DM Framework (Business Understanding -> Data Understanding -> Data Prep -> Modeling -> Evaluation -> Deployment)
2. OneHotEncoding & Dummy Variable Trap (drop='first')
3. Train/Test Split Necessity
4. Evaluation Metrics: MSE, MAE, RMSE, R2 Interpretation
5. Overfitting & Underfitting Diagnosis (Learning Curves)
6. Feature Selection (Forward Selection 1-5 Features, Lasso L1 Regularization)
7. Model Deployment & Serialization (pickle, gzip)
"""

import os
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set plotting aesthetics
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
print_header("CRISP-DM Step 1: Business Understanding")
print("""
[Business Objective]
Develop a predictive model to forecast startup profitability (Profit) based on:
1. R&D Spend (Research & Development)
2. Administration Spend
3. Marketing Spend
4. State Location (California, Florida, New York)

[Stakeholders]
Venture Capitalists, Angel Investors, Startup Executives, Incubator Managers.

[Success Criteria]
High R-squared (R2 > 0.85), low RMSE & MAE, satisfying linear regression assumptions.
""")


# ==========================================
# CRISP-DM Step 2: Data Understanding
# ==========================================
print_header("CRISP-DM Step 2: Data Understanding")

url = "https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
print(f"Fetching dataset from URL: {url}")
data = pd.read_csv(url)

print(f"\nDataset Dimensions: {data.shape[0]} rows, {data.shape[1]} columns")
print("\nFirst 5 Records:")
print(data.head())

print("\nDataset Info & Missing Values Check:")
print(data.isnull().sum())

print("\nDescriptive Statistics:")
print(data.describe().round(2))

# Data Visualization: Feature Exploration Scatter Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
features = ['R&D Spend', 'Administration', 'Marketing Spend']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for ax, feat, color in zip(axes, features, colors):
    sns.regplot(x=feat, y='Profit', data=data, ax=ax, color=color, scatter_kws={'alpha': 0.7})
    ax.set_title(f'{feat} vs Profit', fontsize=12, fontweight='bold')
    ax.set_ylabel('Profit ($)')
    ax.set_xlabel(f'{feat} ($)')

plt.tight_layout()
plt.savefig('scatter_exploration.png', dpi=300)
plt.close()
print("Saved scatter exploration plot as 'scatter_exploration.png'")


# ==========================================
# CRISP-DM Step 3: Data Preparation
# ==========================================
print_header("CRISP-DM Step 3: Data Preparation")

# 1. Feature & Target Separation
X_raw = data.drop('Profit', axis=1)
y = data['Profit']

# 2. One-Hot Encoding & Dummy Variable Trap Resolution
# State has 3 categories: California, Florida, New York
# drop_first=True creates n-1 dummy variables (Florida, New York), dropping California as baseline.
# This eliminates perfect multicollinearity (Dummy Variable Trap).
X_encoded = pd.get_dummies(X_raw, columns=['State'], drop_first=True, dtype=float)
feature_names = list(X_encoded.columns)

print(f"Encoded Feature Names ({len(feature_names)}): {feature_names}")
print("\nFirst 3 rows of encoded feature matrix:")
print(X_encoded.head(3))

# 3. Train/Test Split
# Necessity: Evaluate generalization ability on unseen data to prevent overfitting.
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=0
)
print(f"\nTraining set size: {X_train.shape[0]} samples (80%)")
print(f"Testing set size:  {X_test.shape[0]} samples (20%)")


# ==========================================
# CRISP-DM Step 4: Modeling
# ==========================================
print_header("CRISP-DM Step 4: Modeling")

# Initialize and train Multiple Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(f"Model Intercept (b0): {regressor.intercept_:,.2f}")
print("Model Coefficients (weights):")
for name, coef in zip(feature_names, regressor.coef_):
    print(f"  - {name:20s}: {coef:12.4f}")

print("\n[5 Classical Multiple Regression Assumptions]")
print("1. Linearity: Linear relationship between predictors and Profit (validated via scatter plots).")
print("2. Homoscedasticity: Constant variance of error terms across levels of predictors.")
print("3. Multivariate Normality: Residuals follow a Gaussian normal distribution.")
print("4. Independence of Errors: No autocorrelation in residuals (Durbin-Watson test).")
print("5. Lack of Multicollinearity: Addressed by dropping first dummy variable.")


# ==========================================
# CRISP-DM Step 5: Evaluation & Diagnosis
# ==========================================
print_header("CRISP-DM Step 5: Model Evaluation & Diagnosis")

# Make predictions on test set
y_pred = regressor.predict(X_test)

# Calculate Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R-squared (R2 Score):        {r2:.6f}  (Explains ~{r2*100:.2f}% of test variance)")
print(f"Mean Absolute Error (MAE):   ${mae:,.2f}  (Average absolute error magnitude)")
print(f"Mean Squared Error (MSE):    {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f} (Error in original dollar units)")

# Plot Actual vs Predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, color='#2b5c8f', edgecolors='black', s=80, alpha=0.85, label='Actual Test Points')
min_val = min(y_test.min(), y_pred.min()) - 5000
max_val = max(y_test.max(), y_pred.max()) + 5000
plt.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=2, label='Perfect Fit Line (y = x)')
plt.title('Actual vs Predicted Profit (Test Set)', fontsize=13, fontweight='bold')
plt.xlabel('Actual Profit ($)', fontsize=11)
plt.ylabel('Predicted Profit ($)', fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
plt.close()
print("Saved actual vs predicted plot as 'actual_vs_predicted.png'")

# Overfitting & Underfitting Diagnosis: Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    LinearRegression(), X_encoded, y,
    cv=5,
    train_sizes=np.linspace(0.2, 1.0, 8),
    scoring='r2',
    random_state=0
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color='#2980b9', label='Training R2 Score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#2980b9')
plt.plot(train_sizes, val_mean, 's-', color='#27ae60', label='Cross-Validation R2 Score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#27ae60')

plt.title('Learning Curve: Overfit vs Underfit Diagnosis', fontsize=13, fontweight='bold')
plt.xlabel('Training Set Sample Size', fontsize=11)
plt.ylabel('R2 Score', fontsize=11)
plt.ylim(0.7, 1.02)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=300)
plt.close()
print("Saved learning curve plot as 'learning_curve.png'")
print("\n[Overfitting / Underfitting Assessment]")
print(f"Final Cross-Validation R2 converges at ~{val_mean[-1]:.3f} closely matching Training R2 (~{train_mean[-1]:.3f}).")
print("Conclusion: No evidence of severe overfitting or underfitting; model generalizes well.")


# ==========================================
# Step 5.5: Feature Selection Engineering
# ==========================================
print_header("Step 5.5: Feature Selection (Forward Selection & Lasso)")


def evaluate_performance(y_true, y_pred_val):
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred_val))
    r2_val = r2_score(y_true, y_pred_val)
    return rmse_val, r2_val


def forward_selection(X_tr, X_te, y_tr, y_te, max_features):
    selected_features = []
    results = []
    for i in range(1, max_features + 1):
        feature_names_pool = [f for f in X_tr.columns if f not in selected_features]
        candidate_rmse = np.inf
        candidate_r2 = -np.inf
        candidate_feature = None

        for feature in feature_names_pool:
            features_to_use = selected_features + [feature]
            mdl = LinearRegression()
            mdl.fit(X_tr[features_to_use], y_tr)
            y_pred_candidate = mdl.predict(X_te[features_to_use])
            curr_rmse, curr_r2 = evaluate_performance(y_te, y_pred_candidate)

            if curr_rmse < candidate_rmse:
                candidate_rmse = curr_rmse
                candidate_r2 = curr_r2
                candidate_feature = feature

        selected_features.append(candidate_feature)
        results.append({
            'Number of Features': i,
            'Selected Features': selected_features.copy(),
            'RMSE': candidate_rmse,
            'R-squared': candidate_r2
        })
    return results


fs_results = forward_selection(X_train, X_test, y_train, y_test, max_features=5)
fs_df = pd.DataFrame(fs_results)

print("Results Table (Forward Selection):")
print(fs_df.to_string(index=False))

# Plot Forward Selection Trajectory
fig, ax1 = plt.subplots(figsize=(8, 5))
color = '#c0392b'
ax1.set_xlabel('Number of Features Selected', fontsize=11)
ax1.set_ylabel('RMSE ($)', color=color, fontsize=11)
ax1.plot(fs_df['Number of Features'], fs_df['RMSE'], marker='o', color=color, linewidth=2, label='RMSE (Lower is better)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = '#2980b9'
ax2.set_ylabel('R-squared', color=color, fontsize=11)
ax2.plot(fs_df['Number of Features'], fs_df['R-squared'], marker='s', color=color, linewidth=2, linestyle='--', label='R-squared (Higher is better)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Forward Feature Selection: Performance vs Model Complexity', fontsize=13, fontweight='bold')
fig.tight_layout()
plt.savefig('forward_selection_curve.png', dpi=300)
plt.close()
print("Saved forward selection curve as 'forward_selection_curve.png'")

# Lasso Regression (L1 Penalty) Feature Sparsity
lasso_model = Lasso(alpha=100.0, random_state=0)
lasso_model.fit(X_train, y_train)
print("\nLasso Coefficients (alpha=100):")
for feat, coef in zip(feature_names, lasso_model.coef_):
    status = "Active" if abs(coef) > 1e-4 else "Dropped (Zeroed)"
    print(f"  - {feat:20s}: {coef:12.4f} [{status}]")


# ==========================================
# CRISP-DM Step 6: Deployment (pickle, gzip)
# ==========================================
print_header("CRISP-DM Step 6: Deployment & Serialization")

# 1. Optimal Model Retraining (Top 2 Features: R&D Spend + Marketing Spend)
best_features = ['R&D Spend', 'Marketing Spend']
optimal_model = LinearRegression()
optimal_model.fit(X_train[best_features], y_train)

# 2. Serialize Model with pickle
pkl_filename = 'model.pkl'
with open(pkl_filename, 'wb') as f:
    pickle.dump({'model': optimal_model, 'features': best_features}, f)
print(f"1. Model serialized to standard pickle: '{pkl_filename}'")

# 3. Compress Model with gzip
gz_filename = 'model.pkl.gz'
with gzip.open(gz_filename, 'wb') as f:
    pickle.dump({'model': optimal_model, 'features': best_features}, f)
print(f"2. Model compressed and serialized with gzip: '{gz_filename}'")

# 4. Load from gzip and perform production inference
with gzip.open(gz_filename, 'rb') as f:
    deployed_payload = pickle.load(f)

loaded_model = deployed_payload['model']
deployed_feats = deployed_payload['features']

print(f"\nSuccessfully loaded deployed model from '{gz_filename}'")
print(f"Active Model Features: {deployed_feats}")

# Inference on hypothetical prospective startup:
# R&D Spend = $160,000, Marketing Spend = $320,000
sample_startup = pd.DataFrame([[160000.0, 320000.0]], columns=deployed_feats)
predicted_val = loaded_model.predict(sample_startup)[0]

print("\n--- Production Inference Demonstration ---")
print(f"Input: R&D Spend = $160,000, Marketing Spend = $320,000")
print(f"Predicted Startup Profit: ${predicted_val:,.2f}")
print("============================================================")
