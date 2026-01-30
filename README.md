# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: HARI RAMESH
RegisterNumber: 25009620


from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])
model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')
model.fit(X, y)
print("Weights:", model.coef_)
print("Bias:", model.intercept_)

y_pred = model.predict(X)


plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Perfect prediction line
plt.show()
*/
```

## Output:
<img width="814" height="566" alt="Screenshot 2026-01-30 140714" src="https://github.com/user-attachments/assets/c3c06430-7688-413f-886c-0e559c0a243e" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
