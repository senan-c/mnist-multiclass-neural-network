from sklearn.datasets import fetch_openml
import numpy as np

def load_data():
    #Loading the MNIST dataset
    features, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    features = features.to_numpy()

    #Normalising and transposing the data
    features = features / 255.0
    features = features.T
    labels = labels.astype(int)

    #Shuffling the data
    shuffle = np.arange(len(features))
    np.random.shuffle(shuffle)
    features = features[shuffle]
    labels = labels[shuffle]

    #Splitting the data into training, validation, and testing
    length = features.shape[1]
    train_end = int(length * 0.6)
    valid_end = int(length * 0.8)

    test_features  = features[:, :train_end]
    test_labels    = labels[:train_end]

    valid_features = features[:, train_end:valid_end]
    valid_labels   = labels[train_end:valid_end]

    train_features = features[:, valid_end:]
    train_labels   = labels[valid_end:]

    return test_features, test_labels, valid_features, valid_labels, train_features, train_labels


class MCNeuralNetwork:
    def __init__(self, nodes):
        self.nodes = nodes
        self.length = len(nodes) - 1
        self.test_features, self.test_labels, self.valid_features, self.valid_labels, self.train_features, self.train_labels = load_data()
        self.weights = []
        self.biases = []

        #Initialising weights and biases
        for i in range(self.length):
            B = np.random.randn(nodes[i + 1], 1)
            self.biases.append(B)

            #Using HE initialisation for use with ReLU
            W = np.random.randn(nodes[i + 1], nodes[i]) * np.sqrt(2 / nodes[i])
            self.weights.append(W)

    @staticmethod
    def cost(y_hat, y, m):
        #Clipping predictions to avoid log(0)
        y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)

        #Categorical cross-entropy loss
        loss = np.sum(y * np.log(y_hat)) / m

        #Returning the negative loss
        return loss * -1
    
    @staticmethod
    def batch_softmax(z):
        #Finding the max value in each batch
        max_z = np.max(z, axis=0, keepdims=True)

        #Subtracting max from each value for numerical stability
        exp_z = np.exp(z - max_z)

        #Calculating softmax for batch processing
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    @staticmethod
    def ReLU(z):
        #ReLU activation function
        return np.maximum(0, z)
    
    def feed_forward(self, A0, length):
        cache = [A0]
        #Calculating the activations for each layer
        for i in range(length):
            if i != length - 1:
                Z = np.dot(self.weights[i], cache[i]) + self.biases[i]
                A = self.ReLU(Z)
                #Storing the activations in a cache for backprop
                cache.append(A)

            else:
                Z = np.dot(self.weights[i], cache[i]) + self.biases[i]
                A_last = self.batch_softmax(Z)

        return A_last, cache
    
    def backprop(self, A0, labels, alpha, m):
        y_hat, cache = self.feed_forward(A0, self.length)

        error = self.cost(y_hat, labels)

        for i in range(self.length, 0, -1):
            if i == self.length:
                #For the output layer, we do not apply any function
                A = y_hat
                A_back = cache[i - 1]
                
                #Calculating the gradient of the cost with respect to the activation of this layer
                dCost_dOut = (1 / m) * (A - labels)
                assert dCost_dOut.shape == (self.nodes[i], m)

                #Calculating the gradient of the output layer with respect to the activation of the previous layer
                dOut_dWeight = A_back
                assert dOut_dWeight.shape == (self.nodes[i - 1], m)

                #Calculating the gradient of the cost with respect to the weights of this layer
                dCost_dWeight = np.dot(dCost_dOut, dOut_dWeight.T)
                assert dCost_dWeight.shape == (self.nodes[i], self.nodes[i - 1])

                #Calculating the gradient of the cost with respect to the biases of this layer
                dCost_dBias = np.sum(dCost_dOut, axis=1, keepdims=True)
                assert dCost_dBias.shape == (self.nodes[i], 1)

                #Calculating the gradient of the cost with respect to the activation of previous layer
                dCost_dA_back = np.dot(self.weights[i - 1].T, dCost_dOut)
                assert dCost_dA_back.shape == (self.nodes[i - 1], m)

                #Updating the weights and biases
                self.weights[i - 1] -= alpha * dCost_dWeight
                self.biases[i - 1] -= alpha * dCost_dBias

            else:
                A = cache[i]

                if i == 1:
                    A_back = A0

                else:
                    A_back = cache[i - 1]

                #Calculating the gradient of the cost with respect to the activation of this layer
                dA_dCost = (A > 0).astype(float)
                dReLU = dCost_dA_back * dA_dCost
                assert dReLU.shape == (self.nodes[i], m)

                #Calculating the gradient of the output layer with respect to the activation of the previous layer
                dOut_dWeight = cache[i - 1]
                dCost_dWeight = np.dot(dReLU, dOut_dWeight.T)
                assert dCost_dWeight.shape == (self.nodes[i], self.nodes[i - 1])

                #Calculating the gradient of the cost with respect to the biases of this layer
                dCost_dBias = np.sum(dReLU, axis=1, keepdims=True)
                assert dCost_dBias.shape == (self.nodes[i], 1)

                dCost_dA_back = np.dot(self.weights[i - 1].T, dReLU)
                assert dCost_dA_back.shape == (self.nodes[i - 1], m)

                #Updating the weights and biases
                self.weights[i - 1] -= alpha * dCost_dWeight
                self.biases[i - 1] -= alpha * dCost_dBias

        return error