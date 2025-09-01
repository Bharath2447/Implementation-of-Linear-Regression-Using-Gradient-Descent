# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required library and read the dataframe

2. Write a function computeCost to generate the cost function.

3. Perform iterations og gradient steps with learning rate.

4. Plot the cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Bharath K
RegisterNumber: 212224230036
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X=np.c_[np.ones(len(X1)), X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:, :-2].values)

print(X)

X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:, -1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)

print('Name: Bharath.K')
print("Register No: 212224230036")

print(X1_Scaled)

print(Y1_Scaled)

theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled =scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
*/
```

## Output:
THETA:

<img width="720" height="150" alt="image" src="https://github.com/user-attachments/assets/a4dea689-c670-4846-af9a-29b0565e3fe9" />

X:

<img width="633" height="781" alt="image" src="https://github.com/user-attachments/assets/8528e280-f839-4600-8f35-def8a3e96043" />

Y:

<img width="431" height="748" alt="image" src="https://github.com/user-attachments/assets/451534f3-ccf2-4bd9-8ce3-32b3864e72ac" />

X1_Scaled:

<img width="735" height="767" alt="image" src="https://github.com/user-attachments/assets/c68b5d9d-0916-4c71-90cf-493ba85da4cf" />

Y1_Scaled:

<img width="565" height="775" alt="image" src="https://github.com/user-attachments/assets/5c53a8c2-49fd-410d-8d52-00f2e2100ef3" />

Predicted value: 

<img width="375" height="47" alt="image" src="https://github.com/user-attachments/assets/025c0ccb-3a10-4a83-b5b7-f674ba7f6a93" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
