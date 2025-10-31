
import numpy as np
import random

class NeuralNet():
    def __init__(self,**args):
        self.activaton_values=[]
        self.weight_matrices=[]
        self.learning_rate=args["learning_rate"]
        self.previous_layer_shape=args["input_shape"]

    def activation_function(self,x):
        """
        Sigmoid activation function.
        TODO :implementing this as a parameterised function to handle different activation functions later.
        """
        return 1/(1+np.exp(-x))
    
    def add_layer(self,number_of_neurons):
        """
        Weight matrix connecting the previous layer to the new layer.
        Shape: (neurons_in_new_layer, neurons_in_previous_layer)
        Each neuron in the new layer has a weight for every neuron in the previous layer.
        """
        weight_matrix = np.random.random(number_of_neurons,self.previous_layer_shape)   
        self.previous_layer_shape=number_of_neurons
        self.weight_matrices.append(weight_matrix)
        return
    
    def feedforward(self,input_data):
        """
        Implementing feedforward pass through the network.
        1. For each layer, compute the input to the layer by multiplying the weight matrix
            with the output from the previous layer.
        2. Apply activation function (e.g., sigmoid) to the layer input to get the layer output.
        3. Store the layer outputs for use in backpropagation.
        4. Return the final output of the network.

        Same could be implemented using dot product over the transpose of weight matrix also
        """
        """
        #LAYER_WISE APPROACH:

        self.layerwise_outputs=[]
        input_for_next_layer=input_data
        for weight_matrix in self.weight_matrices:
            layer_input=np.dot(weight_matrix,input_for_next_layer)
            layer_output=self.activation_function(layer_input)
            self.layer_outputs.append(layer_output)
            input_for_next_layer=layer_output
        """
        
        self.activation_values=[] # this list contains activation of each layer as a list
        
        current_activations=input_data
        self.activation_values.append(current_activations) #storing for use in backpropagation
        
        for i in self.weight_matrices:
            current_activations=np.matmul(current_activations,np.transpose(i)) # this would result in a 1X(number of neurons in the current layer) matrix
            current_activations=self.activation_function(current_activations)
            self.activation_values.append(current_activations)
            input_data=current_activations

    def backpropagate(self,expected_output):
        """
        Implementing backpropagation algorithm to update weights based on error.
        1. Compute the error at the output layer.
        2. For each layer, compute the gradient of the error with respect to the weights.
        3. Update the weights using the computed gradients and learning rate.
        """
        pass

def create_neural_net():
    model=NeuralNet()
    model.add_layer(neurons=6) # first hidden layer
    model.add_layer(neurons=6) # second hidden layer
    model.add_layer(neurons=1) # output layer
    return model

def main():
    
    labels = np.random.randint(0, 2, 100) # binary labels for 100 samples
    data = np.random.uniform(-1, 1, (100, 4)) # 100 samples, 4 features each
    print(type(data),type(labels))
    
    neural_network=create_neural_net()
    training_epochs = 1000
    training_learning_rate = 0.01

    for epoch in range(training_epochs):
        error=0
        for index in range(len(data)): # go through each data point
            neural_network.feedforward()
            # neural_network.backpropagate()
            error+=(labels[index]-neural_network.output_layers[-1][0])**2
        print(f"Epoch {epoch+1}, Cumulative Error : {error} ,Error: {error/len(data)}")

