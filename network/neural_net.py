import numpy as np
import math
import random

def sigmoid(x : float) -> float:
    """Measures activation of a neuron as a real number in [0, 1]"""
    return 1 / (1 + math.exp(-x))

def deriv_sigmoid(x : float) -> float:
    """Calculates the derivative of the sigmoid at a point."""
    v = sigmoid(x)
    return v * (1 - v)    

class Net:
    def __init__(self, layer_format):
        self.activations = np.array([np.zeros((k, 1)) for k in layer_format], dtype=object)
        self.biases = np.array([np.random.randn(k, 1) for k in layer_format[1:]], dtype=object)
        self.weights = np.array([np.random.randn(k, l) for l, k in zip (layer_format[0:-1], layer_format[1:])], dtype=object)
    
    def forward(self):
        """Feeds the input in the first layer forward to the other layers."""
        for L in range (0, len(self.activations) - 1):
            self.activations[L + 1] = np.vectorize(sigmoid)(np.dot(self.weights[L], self.activations[L]) + self.biases[L])
            
    def backpropagate(self, y):
        """Computes the partial derivative of the cost function w.r.t. the biases and weights.
        
        The cost function is 
        .. math:: C(a^{(L)}_0,\ldots,a^{(L)}_k) = \sum_{i = 0}^{k}(y_k - a^{(L)}_k) \text{ where } a^{(L)} \text{ is the activation in the last layer.}
        """
        dc_da = np.array([np.zeros((len(l), 1)) for l in self.activations], dtype=object)
        dc_da[-1] = 2 * (self.activations[-1] - y)
        dc_db = np.array([np.zeros((len(l), 1)) for l in self.activations[1:]], dtype=object)
        dc_dw = np.array([np.zeros((len(k), len(l))) for l, k in zip (self.activations[0:-1], self.activations[1:])], dtype=object)
        for L in range (len(self.activations) - 2, -1, -1): 
            dc_db[L] = np.vectorize(deriv_sigmoid)(np.dot(self.weights[L], self.activations[L]) + self.biases[L]) * dc_da[L + 1]
            dc_dw[L] = np.dot(dc_db[L], self.activations[L].transpose())
            dc_da[L] = np.dot(self.weights[L].transpose(), dc_db[L])
        return dc_db, dc_dw 
       
    def train_stochastic(self, rounds, batch_size, learning_rate, data, labels):
        """Splits the randomly shuffled data into batches of size batch_size and applies train_batch to each batch. 
        Repeats this procedure rounds number of times.
        """
        for i in range(rounds):
            shuffled_indices = [i for i in range(len(data))]
            random.shuffle(shuffled_indices)
            for j in range(len(data) // batch_size): 
                batch_data = [0] * batch_size
                batch_labels = [0] * batch_size
                for k in range(batch_size):
                    batch_data[k] = data[shuffled_indices[k + j * batch_size]]
                    batch_labels[k] = labels[shuffled_indices[k + j * batch_size]]
                self.train_batch(batch_data, batch_labels, learning_rate)
    
    def train_batch(self, data, labels, learning_rate):
        """Updates the weights and biases by a factor (learning_rate) of the average of the partial derivatives (i.e. the results of the backpropagate method).
        """
        sum_db, sum_dw = 0,0
        size = len(data)
        for i in range(size):
            self.activations[0] = np.array([[data[i][k]] for k in range(len(self.activations[0]))])
            self.forward()
            y = np.zeros((len(self.activations[-1]), 1))
            y[labels[i]] = [1]
            db, dw = self.backpropagate(y)
            sum_db += db
            sum_dw += dw
        self.biases = self.biases - (learning_rate / size) * sum_db
        self.weights = self.weights - (learning_rate / size) * sum_dw
    
        
    def test(self, training_data, training_labels):
        """Tests the network on the training data and labels, assuming the labels are scalar values. 
        The output of the neural network is interpreted as the maximum value of the activations of the final layer.
        """
        correct = 0
        size = len(training_data)
        for i in range(size):
            self.activations[0] = np.array([[training_data[i][k]] for k in range(len(self.activations[0]))])
            self.forward()
            if (max(self.activations[-1])[0] == self.activations[-1][training_labels[i]][0]):
                correct += 1
        print("ACCURACY: ", correct/size)
        return correct/size
        
 