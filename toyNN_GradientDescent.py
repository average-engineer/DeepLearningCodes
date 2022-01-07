#%% A simple Gradient Descent Algorithm applied for a Toy Neural Network 
# Only one sigmoid Neuron is present
# There is only one input to the neuron and only one output

#%% Importing Packages

import numpy as np
import matplotlib.pyplot as plt

#%% Training Data for Toy Neural Network
X = [0.5,2.5] # Training Data Inputs
Y = [0.2,0.9] # Training Data Outputs

#%% Lists for storing parameters and losses
wList = []
bList = []
lList = []

#%% Sigmoid Activation Function
def sigmoid(w,b,x):
    return 1/(1 + np.exp(-(w*x + b)))

#%% Loss Function
def loss(w,b):
    loss = 0
    for x,y in zip(X,Y):
        loss = loss + 0.5*(sigmoid(w,b,x) - y)**2
    return loss

#%% Bias Gradient
def grad_b(w,b):
    db = 0
    for x,y in zip(X,Y):
        db = db + (sigmoid(w,b,x) - y)*(sigmoid(w,b,x))*(1 - sigmoid(w,b,x))
    return db

#%% Weight Gradient
def grad_w(w,b):
    dw = 0
    for x,y in zip(X,Y):
        dw = dw + (sigmoid(w,b,x) - y)*(sigmoid(w,b,x))*(1 - sigmoid(w,b,x))*(x)
    return dw


#%% Applying the Gradient Descent Algorithm on Training Data

# Initial Weight and Bias
w = -2
b = -2
# Magnitude of change (eta)
eta = 1
# Max number of epochs / iterations
maxEpochs = 100

for i in range(0,maxEpochs):
    w = w - eta*grad_w(w,b)
    b = b - eta*grad_b(w,b)
    wList.append(w)
    bList.append(b)
    lList.append(loss(w,b))
    
#%% Visualising the loss surfaces in the weight bias plane
fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.contour3D(wList,bList,lList)
plt.plot(lList)
plt.ylabel('Loss')
plt.xlabel('# Epochs')

fig , (ax1,ax2) = plt.subplots(nrows = 2,  ncols = 1)
ax1.plot(wList)
ax1.set_ylabel('Weights')
ax1.set_xlabel('# Epochs')

ax2.plot(bList)
ax2.set_ylabel('Biases')
ax2.set_xlabel('# Epochs')
    
    