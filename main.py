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