import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Sample synthetic dataset
# In real use, replace this with: pd.read_csv('your_dataset.csv')
data = pd.DataFrame({
    'Age': np.random.randint(18, 80, 200),
    'Weight': np.random.randint(50, 100, 200),
    'Height': np.random.randint(150, 200, 200),
    'Cholesterol': np.random.randint(150, 300, 200),
    'HeartRate': np.random.randint(60, 100, 200),
    'SystolicBP': np.random.randint(100, 180, 200)  # Target variable
})

# Feature and target
X = data[['Age', 'Weight', 'Height', 'Cholesterol', 'HeartRate']]
y = data['SystolicBP']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plotting actual vs predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Systolic BP")
plt.ylabel("Predicted Systolic BP")
plt.title("Actual vs Predicted Blood Pressure")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()
