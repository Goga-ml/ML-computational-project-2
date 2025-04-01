import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data was formatted like sh*t so I had to do a lot of manual fixing and had to use ; as a delimeter
df = pd.read_csv('training_data/AirQuality.csv', delimiter=';') #<-- reference the location of data correctly here
#This is just normalization process
df = df.drop(['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'RH', 'AH', 'PT08.S4(NO2)', 'T', 'PT08.S5(O3)'], axis=1)
df = df.dropna()

#PT08.S2(NMHC)
#NOx(GT)
#PT08.S3(NOx)
#NO2(GT)
#^^ These are the pollutants you can estimate based on each other's presence
variable_to_pred = 'PT08.S2(NMHC)' #<-- input the pollutant you want to estimate here


#Transforming the data into numpy matrices for processing
y = df[variable_to_pred].head(9000).to_numpy()
y_test = df[variable_to_pred].iloc[9001:9357].to_numpy()
df = df.drop(variable_to_pred, axis=1)
X = df.head(9000).to_numpy()
X_test = df.iloc[9001:9357].to_numpy()


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

# Add labels, title, and legend
plt.title(f'Actual vs Predicted Values for {variable_to_pred}')
plt.xlabel('Index')
plt.ylabel(variable_to_pred)
plt.legend()

# Show the plot
plt.show()
