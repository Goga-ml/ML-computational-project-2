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


df = pd.read_csv('training_data/AirQuality.csv', delimiter=';')
df = df.drop(['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'RH', 'AH', 'PT08.S4(NO2)', 'T', 'PT08.S5(O3)'], axis=1)
df = df.dropna()

#PT08.S2(NMHC)
#NOx(GT)
#PT08.S3(NOx)
#NO2(GT)
#^^ These are the pollutants you can estimate based on each other's presence
variable_to_pred = 'NOx(GT)' #<-- input the pollutant you want to estimate here

y = df[variable_to_pred].head(9000).to_numpy()
y_test = df[variable_to_pred].iloc[9001:9357].to_numpy()
df = df.drop(variable_to_pred, axis=1)
X = df.head(9000).to_numpy()
X_test = df.iloc[9001:9357].to_numpy()

# Add a column of ones to X to account for intercept term
X = np.hstack((np.ones((X.shape[0], 1)), X))
y = y.reshape(-1, 1)
print(X)
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

# Plotting for modified gram
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, alpha=0.7, color='b', label='Actual')
plt.scatter(range(len(y_pred_mgs)), y_pred_mgs.flatten(), alpha=0.7, color='r', label='Predicted (MGS)')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Modified Gram-Schmidt Prediction vs Actual')
plt.legend()
plt.grid()
plt.show()

# Plotting for classic gram
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, alpha=0.7, color='b', label='Actual')
plt.scatter(range(len(y_pred_cgs)), y_pred_cgs.flatten(), alpha=0.7, color='g', label='Predicted (CGS)')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Classic Gram-Schmidt Prediction vs Actual')
plt.legend()
plt.grid()
plt.show()




