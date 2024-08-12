import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__ (self):
        self.layers = []
        
    def passer_layer(self,layer_size):
        #Defines a layer intended to pass any data without modifying
        passer = np.ones(layer_size, dtype=int)
        return np.append(passer, 0).astype(int)
        
    def hidden_layer(self,layer_size,fan_in,entropy=10):
        #Used to connect a layer in both directions
        layer_list = []
        for _ in range(layer_size):
            neuron_weights = np.random.randint(0, entropy, size=fan_in+1)  # Including bias weight
            layer_list.append(neuron_weights)
        return np.array(layer_list)
    
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
    
input_features = np.random.randint(0, 5, size=10)
#Declare Neural Network Class
nn = NeuralNetwork()

#Add a foundation layer
input_layer = nn.passer_layer(8)
#Stack the layer
nn.stack_layer(input_layer)

#Add a fully connected layer
fc1 = nn.hidden_layer(2,8,3)
nn.stack_layer(fc1)

#Add a second fully connected layer
fc2 = nn.hidden_layer(10,2,3)
nn.stack_layer(fc2)

#Pass data through the entire network
print(nn.forward_pass(input_features))
