import numpy as np
import pandas as pd
import math
class NeuralNetwork:
    def __init__ (self):
        self.layers = []
        self.layer = None
    def passer_layer(self,layer_size):
        #Defines a layer intended to pass any data without modifying
        passer = np.ones(layer_size, dtype=int)
        return np.append(passer, 0).astype(int)

    def hidden_layer(self,layer_size,fan_in,entropy=10,pre_weights=None,pre_bias=None):
        #Used to connect a layer in both directions
        if pre_weights is None:
            layer_list = []
            for _ in range(layer_size):
                neuron_weights = np.random.randint(0, entropy, size=fan_in+1)
                layer_list.append(neuron_weights)
            return np.array(layer_list)
        else:
            return np.append(pre_weights, pre_bias)
    
    def activate_layer(self,input_features):
        #Activates the neuron layer
        activation_list = []
        if self.layer.ndim > 1:
            for neuron in self.layer:
                neural_weights = neuron[:len(neuron)-1]
                activation_output = neural_weights * input_features + neuron[-1]
                activation_list.append(sum(activation_output))
            return np.array(activation_list)
        else:
            for i in range(len(self.layer)-1):
                activation_output = self.layer[i] * input_features[i] + self.layer[-1]
                activation_list.append(activation_output)
            return np.array(activation_list)
        
    def set_active_layer(self,layer):
        #Declares current layer activation
        self.layer = layer
    
    def stack_layer(self,layer):
        #adds layer to a stack
        self.layers.append(layer)

    def forward_pass(self,input):
        #Passes forward through the Neural Network Stack
        self.set_active_layer(self.layers[0])
        x = self.activate_layer(input)
        pending_stack = self.layers[1:]
        for i in pending_stack:
            self.set_active_layer(i)
            x = self.activate_layer(x)
        return x   
    
    def ReLU(self):
        negative_position = np.where(self.layer < 0)
        self.layer[negative_position] = 0
        return self.layer

def enclose():
    pass
    # input_features = np.random.randint(0, 5, size=10)
    # pre_weights = [-1,-2,4,8,-7,6,1,-2,8,3]

    # #Declare Neural Network Class
    # nn = NeuralNetwork()

    # #Add a foundation layer
    # input_layer = nn.passer_layer(8)
    # test = nn.hidden_layer(8,2,pre_weights=pre_weights,pre_bias=6)
    # nn.set_active_layer(test)
    # nn.ReLU()

    # #Stack the layer
    # nn.stack_layer(input_layer)
    # print(input_layer)
    # #Add a fully connected layer
    # fc1 = nn.hidden_layer(2,8,3)
    # nn.stack_layer(fc1)
    # print(fc1)
    # #Add a second fully connected layer
    # fc2 = nn.hidden_layer(10,2,3)
    # nn.stack_layer(fc2)
    # print(fc2)
    # #Pass data through the entire network
    # print(nn.forward_pass(input_features))



    # # Initialize values
    # x = 25  # Input value
    # y = 10  # Target Value

    # w = 0.5  # Initial Weight
    # b = 0  # Initial Bias
    # lr = 0.0005

def Loss(output, target):
    return 0.5 * (output - target) ** 2

def forward_pass(input, weight, bias):
    z = weight * input + bias
    return z

def chain(*partial_derivatives):
    partial_derivative = 1
    for derivative in partial_derivatives:
        partial_derivative *= derivative
    return partial_derivative

def new_param(param, lr, dL_dp):
    param = param - lr * dL_dp
    return param


def backprop(x, w, b, y):
    z = forward_pass(x, w, b)
    # Compute loss
    L = Loss(z, y)
    # Derivatives
    dL_dz = z - y
    dz_dw = x 
    dz_db = 1 
    dL_dw = dL_dz * dz_dw 
    dL_db = dL_dz * dz_db
    w = new_param(w, lr, dL_dw)
    b = new_param(b, lr, dL_db)
    return w, b

# # Training loop
# for i in range(10):
#     # Perform backpropagation and update w, b
#     w, b = backprop(x, w, b, y)
    
#     z = forward_pass(x, w, b)
    
#     print(f"Iteration {i+1}: z = {z:.4f}, w = {w:.4f}, b = {b:.4f}")

x = 90
y = 15
lr = 0.1
w1 = 0.5
w2 = 0.8
b1 = 0
b2 = 0
def ReLU(input):
    if input < 0:
        return 0
    else:
        return input
#Forward Pass through the network
z1 = forward_pass(x,w1,b1)
a1 = ReLU(z1)
z2 = forward_pass(a1,w2,b2)
L = 0.5*(z2 - y)**2
print(L)
dL_dz2 = z2 - y
dz2_dw2 = a1
dL_dw2 = dL_dz2 * dz2_dw2
print(dL_dw2)
dL_b2 = dL_dz2
