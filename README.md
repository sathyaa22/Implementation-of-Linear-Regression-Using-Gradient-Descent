# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the program.

Step 2: Add a column to x for the intercept,initialize the theta.

Step 3: Perform graadient descent.

Step 4: Read the csv file.

Step 5: Assuming the last column is ur target variable 'y' and the preceeding column.

Step 6: Learn model parameters.

Step 7: Predict target value for a new data point.

Step 8: Stop the program.

## Program:

Program to implement the linear regression using gradient descent.

Developed by: SATHYAA R

RegisterNumber: 212223100052

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        #Calculation predictions
        predictions=(X).dot(theta).reshape(-1,1)
        
        #calculate rrors
        errors=(predictions-y).reshape(-1,1)
        #Update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("C:/Users/admin/Downloads/50_Startups.csv")
data.head()

#Assuming the last column is your target variavle 'y' and the presede
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#Learn model Parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)
#Predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:

![Screenshot 2024-08-29 093306](https://github.com/user-attachments/assets/c47db437-7e6d-46bc-9939-edca02171368)

Prediction:

![Screenshot 2024-08-29 093458](https://github.com/user-attachments/assets/7449b2bc-cbf3-44f9-93c3-7ed2b796f290)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
