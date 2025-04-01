import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def modified_gram_schmidt_qr(A):
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for k in range(m):
        R[k, k] = np.linalg.norm(A[:, k])
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, m):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= R[k, j] * Q[:, k]
    return Q, R


def classic_gram_schmidt_qr(A):
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for k in range(m):
        Q[:, k] = A[:, k]
        for j in range(k):
            R[j, k] = np.dot(Q[:, j], Q[:, k])
            Q[:, k] -= R[j, k] * Q[:, j]
        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] /= R[k, k]
    return Q, R

#Just correctly reference the data file and run the code
df = pd.read_csv('training_data/car_data.csv')


#dropping non-numeric or redundant data so linear regression works
df = df.drop(['class', 'drive', 'fuel_type', 'make', 'model', 'transmission', 'year'], axis=1)
df = df.dropna()


y = df['combination_mpg'].head(400).to_numpy()
y_test = df['combination_mpg'].iloc[401:550].to_numpy()
df = df.drop('combination_mpg', axis=1)
X = df.head(400).to_numpy()
X_test = df.iloc[401:550].to_numpy()
print(X)

# Add a column of ones to X to account for intercept term
X = np.hstack((np.ones((X.shape[0], 1)), X))
y = y.reshape(-1, 1)

# MGS solution
Q_mgs, R_mgs = modified_gram_schmidt_qr(X)
theta_mgs = np.linalg.solve(R_mgs, Q_mgs.T @ y)

# CGS solution
Q_cgs, R_cgs = classic_gram_schmidt_qr(X)
theta_cgs = np.linalg.solve(R_cgs, Q_cgs.T @ y)

# Saving intercept and coefs from both methods
intercept_mgs, coefficients_mgs = theta_mgs[0, 0], theta_mgs[1:].flatten()
intercept_cgs, coefficients_cgs = theta_cgs[0, 0], theta_cgs[1:].flatten()

# Making predictions on test dataset
X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
y_pred_mgs = X_test_augmented @ theta_mgs
y_pred_cgs = X_test_augmented @ theta_cgs

# Evaluating the different models
mse_mgs = np.mean((y_test.reshape(-1, 1) - y_pred_mgs) ** 2)
mse_cgs = np.mean((y_test.reshape(-1, 1) - y_pred_cgs) ** 2)

ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual_mgs = np.sum((y_test - y_pred_mgs.flatten()) ** 2)
ss_residual_cgs = np.sum((y_test - y_pred_cgs.flatten()) ** 2)

r2_mgs = 1 - (ss_residual_mgs / ss_total)
r2_cgs = 1 - (ss_residual_cgs / ss_total)

# Displaying resuts
print("Modified Gram-Schmidt Intercept:", intercept_mgs)
print("Modified Gram-Schmidt Coefficients:", coefficients_mgs)
print("Modified Gram-Schmidt MSE:", mse_mgs)
print("Modified Gram-Schmidt R-squared:", r2_mgs)

print("\nClassic Gram-Schmidt Intercept:", intercept_cgs)
print("Classic Gram-Schmidt Coefficients:", coefficients_cgs)
print("Classic Gram-Schmidt MSE:", mse_cgs)
print("Classic Gram-Schmidt R-squared:", r2_cgs)

# Plot actual vs. predicted values for MGS
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)  # First subplot for MGS
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual values', alpha=0.6)
plt.scatter(range(len(y_pred_mgs)), y_pred_mgs.flatten(), color='red', label='MGS Predicted values', alpha=0.6)
plt.title('Modified Gram-Schmidt: Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('Combination MPG')
plt.legend()

# Plot actual vs. predicted values for CGS
plt.subplot(1, 2, 2)  # Second subplot for CGS
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual values', alpha=0.6)
plt.scatter(range(len(y_pred_cgs)), y_pred_cgs.flatten(), color='green', label='CGS Predicted values', alpha=0.6)
plt.title('Classic Gram-Schmidt: Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('Combination MPG')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()

