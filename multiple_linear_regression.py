import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# CRISP-DM Step 1: Business Understanding
# ==========================================
# Goal: Predict the "Profit" of a startup based on its spending in 
# R&D, Administration, Marketing, and its geographical State.
# This helps venture capitalists decide whether to invest in a particular startup.
print("--- Step 1: Business Understanding ---")
print("Objective: Predict Startup Profit based on expenditures and location.\n")

# ==========================================
# CRISP-DM Step 2: Data Understanding
# ==========================================
print("--- Step 2: Data Understanding ---")
# Load the dataset
url = "https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
df = pd.read_csv(url)

# Display basic information
print("Data Shape:", df.shape)
print("First 3 rows:\n", df.head(3))
print("\nCheck for missing values:\n", df.isnull().sum())
print("-" * 40)

# ==========================================
# CRISP-DM Step 3: Data Preparation
# ==========================================
print("--- Step 3: Data Preparation ---")
# Split the dataset into Features (X) and Target (y)
# X: R&D Spend, Administration, Marketing Spend, State
# y: Profit
X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values

# Handle Categorical Data: 'State' (Index 3) using One-Hot Encoding
# drop='first' avoids the dummy variable trap (multicollinearity)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split data into Training set (80%) and Test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)
print("-" * 40)

# ==========================================
# CRISP-DM Step 4: Modeling
# ==========================================
print("--- Step 4: Modeling ---")
# Create and train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display the learned coefficients and intercept
print("Model Intercept (b):", round(model.intercept_, 2))
print("Model Coefficients (w):", np.round(model.coef_, 2))
print("-" * 40)

# ==========================================
# CRISP-DM Step 5: Evaluation
# ==========================================
print("--- Step 5: Evaluation ---")
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}") # R2 closer to 1 is better

# Visualization: Actual vs Predicted Profit
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.title('Multiple Linear Regression: Actual vs Predicted Profit')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.legend()
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
print("Plot saved as actual_vs_predicted.png")
try:
    plt.show(block=False)
    plt.pause(0.5)
except Exception:
    pass

# ==========================================
# CRISP-DM Step 6: Deployment
# ==========================================
print("--- Step 6: Deployment ---")
# Let's predict the profit of a new, unseen startup
# Example New Startup: 
# State: California (Encoded as 0, 0 because of drop='first')
# R&D: $150,000, Admin: $100,000, Marketing: $300,000
# Notice the input format matches the transformed X array (Dummy1, Dummy2, R&D, Admin, Marketing)
new_startup = np.array([[0.0, 0.0, 150000.0, 100000.0, 300000.0]])
predicted_profit = model.predict(new_startup)

print(f"Predicted Profit for the new startup: ${predicted_profit[0]:,.2f}")
