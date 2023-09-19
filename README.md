# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression. 

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array. 

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:DHANUMALYA D 
Register Number:212222230030  

```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

df=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=df[:,[0,1]]
y=df[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y) / X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),
                        method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min() - 1,X[:,0].max()+1
  y_min,y_max=X[:,1].min() - 1,X[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))

  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label='admitted')
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label='NOT admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)

```

## Output:
### Array Value of x:
![m51](https://github.com/Dhanudhanaraj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119218812/dd96e726-4869-4eb9-afad-01acedee66cc)

### Array Value of y
![m52](https://github.com/Dhanudhanaraj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119218812/adcfaba5-1887-4021-951c-ed782774aed0)

### Exam 1 - score graph
![m53](https://github.com/Dhanudhanaraj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119218812/08f7ce83-b323-4f39-8eb7-8f45404ff990)

### Sigmoid function graph
![m54](https://github.com/Dhanudhanaraj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119218812/0e3907fa-dad7-491c-8dea-93526bd7109e)

### X_train_grad value
![m55](https://github.com/Dhanudhanaraj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119218812/64040a78-1c79-4f0a-b3bf-e5b017dd0319)

### Y_train_grad value
![m56](https://github.com/Dhanudhanaraj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119218812/b3ac2b13-60d8-4e6e-88a7-2030a59c6902)

### Print res.x
![m57](https://github.com/Dhanudhanaraj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119218812/59cee92d-0154-4d7c-9d43-a2dcd0640a3b)

### Decision boundary - graph for exam score
![m58](https://github.com/Dhanudhanaraj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119218812/1463d48d-e9ed-43fe-ac50-81c98c3dda38)

### Proability value
![m59](https://github.com/Dhanudhanaraj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119218812/67a688d4-23e6-495a-a72c-c0ba893e9b82)

### Prediction value of mean
![m60](https://github.com/Dhanudhanaraj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119218812/56bde7fa-dc99-442d-9df7-e50817974538)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

