<img width="788" height="574" alt="{EE31A92A-992A-4DA2-A5C6-8964C3273F29}" src="https://github.com/user-attachments/assets/521ccfc6-5e04-467f-b3f9-06ba613dfefb" /># SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load California housing data, select features and targets, and split into training and testing sets. 
2. Scale both X (features) and Y (targets) using StandardScaler. 
3. Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.
4. Predict on test data, inverse transform the results, and calculate the mean squared error.  

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: HARI RAMESH
RegisterNumber: 25009620

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
X = np.array([
    [800, 2],
    [1000, 3],
    [1200, 3],
    [1500, 4],
    [1800, 4],
    [2000, 5],
    [2200, 5],
    [2500, 6]
])

y = np.array([
    [40, 2],
    [55, 3],
    [65, 3],
    [85, 4],
    [95, 4],
    [110, 5],
    [125, 5],
    [145, 6]
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd = SGDRegressor(
    max_iter=2000,
    eta0=0.01,
    learning_rate='constant',
    random_state=42
)

model = MultiOutputRegressor(sgd)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

print("Predicted [Price, Occupants]:")
print(y_pred)

print("\nActual [Price, Occupants]:")
print(y_test)

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", )
new_house = np.array([[1600, 4]])
new_house_scaled = scaler.transform(new_house)

new_prediction = model.predict(new_house_scaled)

print("\nFor New House [1600 sq ft, 4 rooms]:")
print("Predicted House Price (lakhs):", round(new_prediction[0][0], 2))
print("Predicted Number of Occupants:", round(new_prediction[0][1]))
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X_test[:, 0], y_test[:, 0], label="Actual Price")
plt.scatter(X_test[:, 0], y_pred[:, 0], label="Predicted Price")
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price (lakhs)")
plt.title("House Size vs House Price (Actual vs Predicted)")
plt.legend()
plt.show()
*/
```

## Output:
<img width="814" height="566" alt="Screenshot 2026-01-30 140714" src="https://github.com/user-attachments/assets/c3c06430-7688-413f-886c-0e559c0a243e" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
