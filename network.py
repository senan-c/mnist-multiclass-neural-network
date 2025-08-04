from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import time

def one_hot_encode(labels, num_classes=10):
    #Data must be formatted to work with batch processing
    m = labels.shape[0]
    #Creating a one-hot encoded 2D array
    one_hot = np.zeros((m, num_classes))
    #Setting the appropriate indices to 1
    one_hot[np.arange(m), labels] = 1
    return one_hot

def load_data():
    #Loading the MNIST dataset
    features, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    features = features.to_numpy()
    labels = labels.to_numpy()

    #Normalising and transposing the data
    features = features / 255.0
    labels = labels.astype(int)

    #Shuffling the data
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    #Splitting the data into training, validation, and testing
    length = features.shape[0]
    train_end = int(length * 0.6)
    valid_end = int(length * 0.8)

    train_features = features[:train_end, :]
    train_labels = labels[:train_end]

    valid_features = features[train_end:valid_end, :]
    valid_labels = labels[train_end:valid_end]

    test_features = features[valid_end:, :]
    test_labels = labels[valid_end:]

    #One-hot encoding the labels
    test_labels = one_hot_encode(test_labels)
    valid_labels = one_hot_encode(valid_labels)
    train_labels = one_hot_encode(train_labels)

    return test_features, test_labels, valid_features, valid_labels, train_features, train_labels


class MCNeuralNetwork:
    def __init__(self, nodes, test_features, test_labels, valid_features, valid_labels, train_features, train_labels):
        self.nodes = nodes
        self.length = len(nodes) - 1
        self.test_features = test_features
        self.test_labels = test_labels
        self.valid_features = valid_features
        self.valid_labels = valid_labels
        self.train_features = train_features
        self.train_labels = train_labels
        self.weights = []
        self.biases = []

        #Initialising weights and biases
        for i in range(self.length):
            B = np.zeros((1, self.nodes[i + 1]))
            self.biases.append(B)

            #Using HE initialisation for use with ReLU
            W = np.random.randn(self.nodes[i], self.nodes[i+1]) * np.sqrt(2 / self.nodes[i])
            self.weights.append(W)

    @staticmethod
    def cost(y_hat, y):
        #Clipping predictions to avoid log(0)
        y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)

        #Categorical cross-entropy loss
        loss = np.mean(np.sum(y * np.log(y_hat), axis=1))

        #Returning the negative loss
        return loss * -1
    
    @staticmethod
    def batch_softmax(z):
        #Finding the max value in each batch
        max_z = np.max(z, axis=1, keepdims=True)

        #Subtracting max from each value for numerical stability
        exp_z = np.exp(z - max_z)

        #Calculating softmax for batch processing
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def ReLU(z):
        #ReLU activation function
        return np.maximum(0, z)
    
    def feed_forward(self, A0, length):
        cache = [A0]
        #Calculating the activations for each layer
        for i in range(length):
            if i != length - 1:
                Z = np.dot(cache[i], self.weights[i]) + self.biases[i]
                A = self.ReLU(Z)
                #Storing the activations in a cache for backprop
                cache.append(A)

            else:
                Z = np.dot(cache[i], self.weights[i]) + self.biases[i]
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

                #Calculating the gradient of the output layer with respect to the activation of the previous layer
                dOut_dWeight = A_back

                #Calculating the gradient of the cost with respect to the weights of this layer
                dCost_dWeight = np.dot(dOut_dWeight.T, dCost_dOut)

                #Calculating the gradient of the cost with respect to the biases of this layer
                dCost_dBias = np.sum(dCost_dOut, axis=0, keepdims=True)

                #Calculating the gradient of the cost with respect to the activation of previous layer
                dCost_dA_back = np.dot(dCost_dOut, self.weights[i - 1].T)

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

                #Calculating the gradient of the output layer with respect to the activation of the previous layer
                dOut_dWeight = cache[i - 1]
                dCost_dWeight = np.dot(dOut_dWeight.T, dReLU)

                #Calculating the gradient of the cost with respect to the biases of this layer
                dCost_dBias = np.sum(dReLU, axis=0, keepdims=True)

                dCost_dA_back = np.dot(dReLU, self.weights[i - 1].T)

                #Updating the weights and biases
                self.weights[i - 1] -= alpha * dCost_dWeight
                self.biases[i - 1] -= alpha * dCost_dBias

        return error
    
    def accuracy(self, features, labels):
        y_hat, _ = self.feed_forward(features, self.length)
        #Using argmax to find the predicted class
        outputs = np.argmax(y_hat, axis=1)

        #Finding the correct class from the one-hot encoded labels
        correct = np.argmax(labels, axis=1)

        #Calculating the accuracy by comparing the predicted and correct classes
        return np.mean(outputs == correct)
    
    def show_misclassified(self, num=3):
        y_hat, _ = self.feed_forward(self.test_features, self.length)
        #Using argmax to find the predicted class
        predictions = np.argmax(y_hat, axis=1)

        #Finding the actual classes from the one-hot encoded labels
        actuals = np.argmax(self.test_labels, axis=1)

        #Finding the indices of misclassified samples
        wrong = np.where(predictions != actuals)[0]

        for i in wrong[:num]:
            #Displaying the misclassified samples
            plt.imshow(self.test_features[i].reshape(28, 28), cmap='gray')
            plt.title(f"Predicted: {predictions[i]}, Actual: {actuals[i]}")
            plt.axis('off')
            plt.show()

    def reload_data(self, test_features, test_labels, valid_features, valid_labels, train_features, train_labels    ):
        self.test_features = test_features
        self.test_labels = test_labels
        self.valid_features = valid_features
        self.valid_labels = valid_labels
        self.train_features = train_features
        self.train_labels = train_labels
    
    def train(self, epochs, alpha, batch_size):
        m = self.train_features.shape[0]
        total_samples = 0
        costs = []
        accuracies = []

        for e in range(epochs + 1):
            #Getting the start time each epoch
            epoch_start = time.time()

            #Shuffling the training data order for batch processing
            indices = np.arange(m)
            np.random.shuffle(indices)

            total_cost = 0
            num_batches = 0

            for b in range(0, m, batch_size):
                #Setting the end of the batch and clipping it
                end = min(b + batch_size, m)
                #Calculating the total samples processed
                total_samples += end - b

                #Picks a random batch using the shuffled indices
                batch_indices = indices[b:end]

                #Selecting the features and labels for the batch
                A0 = self.train_features[batch_indices, :]
                labels = self.train_labels[batch_indices]

                error = self.backprop(A0, labels, alpha, A0.shape[0])

                total_cost += error
                num_batches += 1

            #Calculating the average cost and accuracy for the epoch
            avg_cost = total_cost / num_batches
            val_acc = self.accuracy(self.valid_features, self.valid_labels)
            train_acc = self.accuracy(self.train_features, self.train_labels)

            if e % 5 == 0:
                #Printing every 5 epochs after the first
                elapsed = time.time() - epoch_start
                print(f"Epoch {e:02}/{epochs} - Cost: {round(avg_cost, 6):.6f} - Val Acc: {val_acc * 100:.4f}% - Train Acc: {train_acc * 100:.4f}% - Time: {elapsed:.2f}s")

            costs.append(avg_cost)
            accuracies.append(train_acc)

        print("\nTotal Samples Processed:", total_samples)

        #Plotting the training cost and accuracy over epochs on a line graph
        plt.plot(range(epochs + 1), costs, label="Cost")
        plt.plot(range(epochs + 1), accuracies, label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Training Cost and Accuracy over Epochs")
        plt.grid(True)
        plt.show()

        self.show_misclassified()

    def test(self):
        #Calculating the accuracy on the test set
        test_acc = self.accuracy(self.test_features, self.test_labels)
        print(f"Test Accuracy: {test_acc * 100:.4f}%")
