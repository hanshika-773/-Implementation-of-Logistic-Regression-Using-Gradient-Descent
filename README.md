# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Probability Prediction: It predicts the probability of a binary outcome (0 or 1) using the sigmoid function.
2. Loss Minimization: The algorithm learns by minimizing a loss function (like cross-entropy) that measures the error between predictions and actual values.
3. Gradient Descent Optimization: It uses gradient descent to iteratively adjust the model's parameters to reduce the loss.
4. Decision Boundary: The learned parameters define a decision boundary that separates the two classes.
5. Classification: By applying a threshold (e.g., 0.5) to the predicted probabilities, it classifies data points into one of the two classes.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: HAnshika Varthini R  
RegisterNumber:  212223240046

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("Placement_Data.csv")
data.info()
data=data.drop(['sl_no','salary'],axis=1)
data
data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data.dtypes
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data
x=data.iloc[:,:-1].values
y=data.iloc[:,-1]
theta = np.random.randn(x.shape[1])
    
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def loss(theta, x, y):
    h = sigmoid(x.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(x.dot(theta))
        gradient = x.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta  
    
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)
    
def predict(theta, x):
    h = sigmoid(x.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
    
y_pred = predict(theta, x)
    
accuracy=np.mean(y_pred.flatten()==y)
    
print("Acuracy:",accuracy)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,5,65,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
*/
```

## Output:
![Screenshot 2025-04-28 114238](https://github.com/user-attachments/assets/c0c43e52-0cd2-46d4-ae69-17f1e7071727)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

