import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        # initialize neural network
        self.learning_rate = learning_rate
        self.weights = [] 
        self.biases = []   
        
        # fefine the full layer structure, including input, hidden, and output layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            # weights are initialized with small random values scaled by sqrt(2 / layer size)
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i+1]))  # biases are initialized to zero
            self.weights.append(weight)  # add weights for this layer
            self.biases.append(bias)     # add biases for this layer
        
    def relu(self, z): # apply ReLU activation function to the input z
        return np.maximum(0, z)
    
    def relu_derivative(self, z): # compute the derivative of the ReLU function (used for backpropagation)
        return (z > 0).astype(float)
    
    def sigmoid(self, z): # apply Sigmoid activation function to the input z (used for the output layer in classification)
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X): # perform a forward pass through the network, calculating activations for each layer
        activations = [X]  # list to store activations of each layer, starting with input X
        zs = []            # list to store the linear transformations before activation
        
        # loop through each layer except the output layer
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]  # linear transformation
            zs.append(z)  # store linear transformation (z) for backpropagation
            a = self.relu(z)  # apply ReLU activation for hidden layers
            activations.append(a)  # store the activated output
        
        # output layer: apply Sigmoid activation for classification
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]  # linear transformation for output layer
        zs.append(z)
        a = self.sigmoid(z)
        activations.append(a)  # store output layer activation
        
        return activations, zs
    
    def backward(self, X, y, activations, zs): # perform backpropagation to compute gradients and update weights and biases
        m = X.shape[0]  # number of examples in the batch
        
        # compute initial error at output layer
        delta = (activations[-1] - y)
        deltas = [delta]  # initialize list of deltas, starting with output layer error
        
        # backpropagate the error through each hidden layer
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T) * self.relu_derivative(zs[l])  # calculate delta for layer
            deltas.insert(0, delta)  # insert delta at the beginning of the list
        
        # update weights and biases using computed deltas and learning rate
        for i in range(len(self.weights)):
            # update weights by gradient descent: subtract learning rate times gradient
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / m
            # update biases by gradient descent: subtract learning rate times average delta
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
        
    def fit(self, X, y, epochs=10, batch_size=64):
        # train the neural network using mini-batch gradient descent
        m = X.shape[0]  # number of training examples
        
        for epoch in range(epochs):
            # shuffle data at each epoch to create mini-batches
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            # loop over each mini-batch
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]  # select batch of inputs
                y_batch = y_shuffled[i:i+batch_size]  # select batch of labels
                activations, zs = self.forward(X_batch)  # perform forward pass on the batch
                self.backward(X_batch, y_batch, activations, zs)  # perform backward pass on the batch
            
            # compute accuracy on the entire dataset at the end of each epoch
            if epoch % 1 == 0:
                y_pred = self.predict(X)
                accuracy = np.mean((y_pred > 0.5) == y)  # accuracy threshold at 0.5 for classification
                print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy * 100:.2f}%")
        
    def predict(self, X):
        # make predictions by performing a forward pass and returning the output layer activations
        activations, _ = self.forward(X)
        return activations[-1] 
