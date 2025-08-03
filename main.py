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

