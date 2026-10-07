import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. Load and Prepare Data
# ==========================================
print("==================================================")
print("  5 FEATURE SELECTION METHODS COMPARISON PIPELINE")
print("==================================================\n")

url = "https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
df = pd.read_csv(url)

print(f"Dataset shape: {df.shape}")
print(df.head(3))
print("\n" + "="*50)

# One-hot encode categorical feature 'State' (drop_first=True to avoid dummy variable trap)
df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True, dtype=float)

# Separate features (X) and target (y)
feature_cols = [col for col in df_encoded.columns if col != 'Profit']
X = df_encoded[feature_cols]
y = df_encoded['Profit']

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Target: Profit\n")

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardized features (essential for Lasso / distance-sensitive methods)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

# Dictionary to hold feature importance scores and ranks
comparison_dict = {'Feature': feature_cols}

# ==========================================
# Method 1: Pearson Correlation Coefficient (Filter)
# ==========================================
print("--- [Method 1] Pearson Correlation Coefficient (Filter) ---")
# Compute Pearson correlation with target variable Profit
corr_matrix = df_encoded.corr(method='pearson')
pearson_corr = corr_matrix['Profit'].drop('Profit')
pearson_abs = pearson_corr.abs()

comparison_dict['Pearson_Corr'] = pearson_corr.values
comparison_dict['Pearson_Abs'] = pearson_abs.values
# Rank 1 = highest absolute correlation
comparison_dict['Pearson_Rank'] = pearson_abs.rank(ascending=False).astype(int).values

for feat, r, rank in zip(feature_cols, pearson_corr, comparison_dict['Pearson_Rank']):
    print(f"  - {feat:20s}: r = {r:+.4f} (Rank {rank})")
print()

# ==========================================
# Method 2: Mutual Information (Filter)
# ==========================================
print("--- [Method 2] Mutual Information (Filter) ---")
# Calculates non-linear & linear dependency reduction of uncertainty (entropy)
mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=feature_cols)

comparison_dict['Mutual_Info'] = mi_scores
comparison_dict['MI_Rank'] = mi_series.rank(ascending=False).astype(int).values

for feat, mi, rank in zip(feature_cols, mi_scores, comparison_dict['MI_Rank']):
    print(f"  - {feat:20s}: MI = {mi:.4f} (Rank {rank})")
print()

# ==========================================
# Method 3: Recursive Feature Elimination / RFE (Wrapper)
# ==========================================
print("--- [Method 3] Recursive Feature Elimination / RFE (Wrapper) ---")
# Iteratively removes weakest features using Linear Regression
estimator = LinearRegression()
rfe = RFE(estimator=estimator, n_features_to_select=1)  # rank all features down to 1
rfe.fit(X_train_scaled, y_train)

comparison_dict['RFE_Rank'] = rfe.ranking_
# Inverse score for visualization (5 is top, 1 is lowest rank)
comparison_dict['RFE_Score'] = (len(feature_cols) + 1) - rfe.ranking_

for feat, rank in zip(feature_cols, rfe.ranking_):
    print(f"  - {feat:20s}: RFE Rank = {rank}")
print()

# ==========================================
# Method 4: L1 Regularization / Lasso Regression (Embedded)
# ==========================================
print("--- [Method 4] L1 Regularization / Lasso (Embedded) ---")
# Lasso with 5-fold cross-validation to find optimal alpha on standardized features
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)
lasso_coefs = lasso.coef_
lasso_abs = np.abs(lasso_coefs)
lasso_series = pd.Series(lasso_abs, index=feature_cols)

comparison_dict['Lasso_Coef'] = lasso_coefs
comparison_dict['Lasso_Abs'] = lasso_abs
comparison_dict['Lasso_Rank'] = lasso_series.rank(ascending=False, method='min').astype(int).values

print(f"  Optimal Alpha: {lasso.alpha_:.4f}")
for feat, coef, rank in zip(feature_cols, lasso_coefs, comparison_dict['Lasso_Rank']):
    print(f"  - {feat:20s}: Coef = {coef:12.2f} (Rank {rank})")
print()

# ==========================================
# Method 5: Random Forest Feature Importance (Embedded)
# ==========================================
print("--- [Method 5] Random Forest Feature Importance (Embedded) ---")
# Calculates mean decrease in impurity (MDI) across all decision trees
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_importances = rf.feature_importances_
rf_series = pd.Series(rf_importances, index=feature_cols)

comparison_dict['RF_Importance'] = rf_importances
comparison_dict['RF_Rank'] = rf_series.rank(ascending=False).astype(int).values

for feat, imp, rank in zip(feature_cols, rf_importances, comparison_dict['RF_Rank']):
    print(f"  - {feat:20s}: Importance = {imp:.4f} ({imp*100:5.2f}%) (Rank {rank})")
print("\n" + "="*50)

# ==========================================
# Summary DataFrame
# ==========================================
df_summary = pd.DataFrame(comparison_dict)

# Calculate Mean Rank across all 5 methods
rank_columns = ['Pearson_Rank', 'MI_Rank', 'RFE_Rank', 'Lasso_Rank', 'RF_Rank']
df_summary['Average_Rank'] = df_summary[rank_columns].mean(axis=1)
df_summary = df_summary.sort_values(by='Average_Rank').reset_index(drop=True)

print("\n--- FEATURE SELECTION COMPARISON SUMMARY TABLE ---")
display_cols = ['Feature', 'Pearson_Rank', 'MI_Rank', 'RFE_Rank', 'Lasso_Rank', 'RF_Rank', 'Average_Rank']
print(df_summary[display_cols].to_string(index=False))
print("\n" + "="*50)

# ==========================================
# Model Performance Evaluation with Feature Subsets
# ==========================================
print("\n--- Model Performance Comparison (Top 1, Top 2, Top 3 vs All Features) ---")
results = []

# Baseline: All 5 Features
base_lr = LinearRegression()
base_lr.fit(X_train_scaled, y_train)
y_pred_all = base_lr.predict(X_test_scaled)
results.append({
    'Feature Subset': 'All Features (5)',
    'Features Used': ', '.join(feature_cols),
    'R2 Score': r2_score(y_test, y_pred_all),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_all)),
    'MAE': mean_absolute_error(y_test, y_pred_all)
})

# Test top 1, 2, 3 ranked features based on Consensus (Average Rank)
top_features = df_summary['Feature'].tolist()
for k in [1, 2, 3]:
    selected_k = top_features[:k]
    sub_lr = LinearRegression()
    sub_lr.fit(X_train_scaled[selected_k], y_train)
    y_pred_k = sub_lr.predict(X_test_scaled[selected_k])
    results.append({
        'Feature Subset': f'Top {k} by Consensus',
        'Features Used': ', '.join(selected_k),
        'R2 Score': r2_score(y_test, y_pred_k),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_k)),
        'MAE': mean_absolute_error(y_test, y_pred_k)
    })

df_perf = pd.DataFrame(results)
print(df_perf[['Feature Subset', 'Features Used', 'R2 Score', 'RMSE', 'MAE']].to_string(index=False))

# ==========================================
# Visualizations
# ==========================================
fig = plt.figure(figsize=(18, 12))

# Subplot 1: Pearson Correlation
ax1 = plt.subplot(2, 3, 1)
df_p = df_summary.sort_values('Pearson_Abs', ascending=False)
sns.barplot(x='Pearson_Abs', y='Feature', data=df_p, hue='Feature', legend=False, ax=ax1, palette='Blues_r')
ax1.set_title('1. Pearson |Correlation| (Filter)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Absolute Pearson Correlation |r|')
for p in ax1.patches:
    ax1.annotate(f"{p.get_width():.3f}", (p.get_width() + 0.02, p.get_y() + p.get_height()/2), va='center', fontsize=10)

# Subplot 2: Mutual Information
ax2 = plt.subplot(2, 3, 2)
df_mi = df_summary.sort_values('Mutual_Info', ascending=False)
sns.barplot(x='Mutual_Info', y='Feature', data=df_mi, hue='Feature', legend=False, ax=ax2, palette='Greens_r')
ax2.set_title('2. Mutual Information (Filter)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Mutual Information Score (Nats)')
for p in ax2.patches:
    ax2.annotate(f"{p.get_width():.3f}", (p.get_width() + 0.01, p.get_y() + p.get_height()/2), va='center', fontsize=10)

# Subplot 3: RFE Importance Score
ax3 = plt.subplot(2, 3, 3)
df_rfe = df_summary.sort_values('RFE_Score', ascending=False)
sns.barplot(x='RFE_Score', y='Feature', data=df_rfe, hue='Feature', legend=False, ax=ax3, palette='Oranges_r')
ax3.set_title('3. RFE Ranking Score (Wrapper)', fontsize=13, fontweight='bold')
ax3.set_xlabel('Ranking Score (Higher = Selected Earlier)')
for p in ax3.patches:
    ax3.annotate(f"Rank {int(len(feature_cols) + 1 - p.get_width())}", (p.get_width() + 0.1, p.get_y() + p.get_height()/2), va='center', fontsize=10)

# Subplot 4: Lasso Absolute Coefficients
ax4 = plt.subplot(2, 3, 4)
df_l = df_summary.sort_values('Lasso_Abs', ascending=False)
sns.barplot(x='Lasso_Abs', y='Feature', data=df_l, hue='Feature', legend=False, ax=ax4, palette='Reds_r')
ax4.set_title('4. Lasso |Coefficients| (Embedded)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Absolute Standardized Coefficient')
for p in ax4.patches:
    ax4.annotate(f"{p.get_width():.0f}", (p.get_width() + 500, p.get_y() + p.get_height()/2), va='center', fontsize=10)

# Subplot 5: Random Forest Importance
ax5 = plt.subplot(2, 3, 5)
df_rf = df_summary.sort_values('RF_Importance', ascending=False)
sns.barplot(x='RF_Importance', y='Feature', data=df_rf, hue='Feature', legend=False, ax=ax5, palette='Purples_r')
ax5.set_title('5. Random Forest MDI (Embedded)', fontsize=13, fontweight='bold')
ax5.set_xlabel('Gini / Impurity Reduction Ratio')
for p in ax5.patches:
    ax5.annotate(f"{p.get_width()*100:.1f}%", (p.get_width() + 0.02, p.get_y() + p.get_height()/2), va='center', fontsize=10)

# Subplot 6: Ranking Comparison Heatmap
ax6 = plt.subplot(2, 3, 6)
heatmap_data = df_summary.set_index('Feature')[rank_columns]
heatmap_data.columns = ['Pearson', 'MI', 'RFE', 'Lasso', 'RF']
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu_r', cbar=True, fmt='d', linewidths=1, ax=ax6)
ax6.set_title('Overall Rank Comparison (1 = Top Feature)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_selection_comparison.png', dpi=300)
print("\nPlot saved successfully to feature_selection_comparison.png")

# Also generate a dedicated Model Performance Comparison plot
fig2, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(df_perf['Feature Subset'], df_perf['R2 Score'], color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], width=0.55)
ax.set_ylim(0.85, 0.95)
ax.set_ylabel('$R^2$ Score on Test Set', fontsize=11)
ax.set_title('Test $R^2$ Score with Selected Feature Subsets', fontsize=13, fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300)
print("Model performance plot saved to model_performance_comparison.png")
