#%% Only one neural network dealing with both u and f
import tensorflow as tf
import numpy as np
from tensorflow import keras
import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer # For creating a custom layer
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from numpy.random import seed
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K # Keras Backend
from tensorflow.python.ops import math_ops # Tensorflow math operations

#%%
L = 100
nData = 32

# Constructing Training Data
# Collocation Points are assumed to be linearly distributed over the bar
xTrain = np.linspace(0,L,nData)
print(xTrain)
xTrain = xTrain.reshape(nData,1,1)

#%% Neural Network (u(x))
# Exact Solution for the displacement of the bar
u_exact = -(0.000011/6)*(xTrain**3) + 0.055*xTrain
#print(u_exact)
u_exact.reshape((nData,1))

uTrain = tf.convert_to_tensor(u_exact,dtype = tf.float32)
xTrain = tf.convert_to_tensor(xTrain,dtype = tf.float32)

#%% Adding a new custom layer for the model paramter lambda
# lambda is introduced as a trainable bias in this layer
class lambdaLayer(Layer):

    # add an activation parameter
    def __init__(self, units, activation=None, name = None): # Default value of neurons = 32, can be changed
        super(lambdaLayer, self).__init__()
        self.units = units
        
        # define the activation to get from the built-in activation layers in Keras
        self.activation = tf.keras.activations.get(activation)


    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel",
            initial_value=w_init(shape=(input_shape[-1], self.units),dtype='float32'),trainable=True)
        b_init = tf.zeros_initializer() 
        self.b = tf.Variable(name="bias",
            initial_value=b_init(shape=(self.units,), dtype='float32'),trainable=True) # shape is equal to the number of neurons in the layer
        #super().build(input_shape)
        lbda_init = tf.zeros_initializer() # Lambda will be the additional trainable parameter
        self.lbda = tf.Variable(name="Lambda",
            initial_value=b_init(shape=(1,), dtype='float32'),trainable=True) 
        super().build(input_shape)


    def call(self, inputs): # Computes the output of the layer based on the trainable parameters and the activation function
        
        # pass the computation to the activation layer
        # print('Input Shape is->')
        # print(type(inputs))
        # print('Weight Shape is->')
        # print(type(self.w))
        return self.activation(tf.matmul(inputs, self.w) + self.b - self.lbda)

#%% Implementing a simple Feed Forward Neural Network
epochs = 100
seed(100)
tf.random.set_seed(1000)
model_uf = Sequential()
# Input Layer
model_uf.add(keras.Input(shape=(1,1))) # nData datapoints of 1x1 data
# Hidden Layers
# Custom Layer for adding lambda as a trainable parameter
cusLayer = lambdaLayer(300,activation = 'tanh', name = 'LambdaLayer')
model_uf.add(layers.Dense(300,activation = 'tanh', name = 'Layer1'))
model_uf.add(layers.Dense(300,activation = 'tanh', name = 'Layer2'))
model_uf.add(layers.Dense(300,activation = 'tanh', name = 'Layer3'))
model_uf.add(cusLayer)
 
# model_uf.add(layers.Dense(300,activation = 'tanh', name = 'Layer4'))
# model_uf.add(layers.Dense(300,activation = 'tanh', name = 'Layer5'))
# model_uf.add(layers.Dense(300,activation = 'tanh', name = 'Layer6'))
# model_uf.add(layers.Dense(300,activation = 'relu', name = 'Layer7'))
# Output Layer
model_uf.add(layers.Dense(1, name = 'OutputLayer'))

# Output from the Neural Network
# uPred = model(xTrain)


#%% Customised Loss Function
# Returns the loss
def lossFunc(x,n,L,lbda):
    # x: Training Data
    # n: Number of datapoints
    # lbda: Model Parameter
    # Variation in x
    eps = 1e-2
    # Loss function taking in only the predicted and exact values
    def loss(uExact,uPred):
        diff1 = math_ops.squared_difference(uExact, uPred) # Squared Difference of the training data
        # Differentiating Model output
        # Finite Difference formulas
        diff_u = (model_uf(x + eps) - model_uf(x - eps))/(2*eps)
        diff2_u = (model_uf(x + eps) - 2*model_uf(x) + model_uf(x - eps))/(eps**2)
        # Approximated PDE Residual
        fPred = diff2_u + (lbda*x)/(L)
        # Squared difference of the PDE Residual
        #diff2 = tf.math.square(fPred)
        diff2 = fPred**2
        return (diff1 + diff2)/(n)
    
    #loss = K.mean(diff,axis = 0) # Mean over first dimension of the tensor (50 datapoints)
    return loss

# Compiling the model
model_uf.compile(loss = lossFunc(xTrain,nData,L,np.array(cusLayer.lbda)[0]), optimizer = 'adam')
model_uf.summary()
history = model_uf.fit(xTrain,uTrain,epochs = epochs)

# Predicted Values
uPred = model_uf.predict(xTrain) # Predicted u(x)
lambdaPred = np.array(cusLayer.lbda) # Predicted lambda (Model Parameter)
print(lambdaPred)

# Saving the trained model 
model_uf.save('saved_model/NN_uf')

#%% Predicted Values
fig = plt.figure(figsize = (6,6))
plt.plot(uPred[:,0,0],color = 'blue', label = 'Predicted')
plt.plot(u_exact[:,0,0],color = 'red', label = 'Exact')
plt.legend()
plt.grid()
plt.show()

fig = plt.figure(figsize = (6,6))
plt.plot(history.history['loss'], label = 'Training Loss')
plt.xlabel('# Epochs')
plt.legend()
plt.grid()


# %%
