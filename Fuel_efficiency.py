import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Just correctly reference the data file and run the code
df = pd.read_csv('training_data/car_data.csv')

#dropping non-numeric or redundant data so linear regression works
df = df.drop(['class', 'drive', 'fuel_type', 'make', 'model', 'transmission', 'year'], axis=1)
df = df.dropna()
print(df.columns)
y = df['combination_mpg'].head(400).to_numpy()
y_test = df['combination_mpg'].iloc[401:550].to_numpy()
df = df.drop('combination_mpg', axis=1)
X = df.head(400).to_numpy()
X_test = df.iloc[401:550].to_numpy()
print(X)

# Add a column of ones to X to account for intercept term
X = np.hstack((np.ones((X.shape[0], 1)), X))
y = y.reshape(-1, 1)

# Calculate coefs using the normal equation
# theta = (X.T * X)^(-1) * X.T * y
X_transpose = X.T
theta = np.linalg.inv(X_transpose @ X) @ (X_transpose @ y)

#Save intercept and coefs
intercept = theta[0, 0]
coefficients = theta[1:].flatten()

# Predict
X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
y_pred = X_test_augmented @ theta

# Evaluating
mse = np.mean((y_test.reshape(-1, 1) - y_pred) ** 2)
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred.flatten()) ** 2)
r2 = 1 - (ss_residual / ss_total)

# Display results
print("Intercept:", intercept)
print("Coefficients:", coefficients)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))

# Plot actual points
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual values', alpha=0.6)

# Plot predicted points
plt.scatter(range(len(y_pred)), y_pred.flatten(), color='red', label='Predicted values', alpha=0.6)

# Adding titles and labels
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Combination MPG')
plt.legend()

# Show plot
plt.show()
