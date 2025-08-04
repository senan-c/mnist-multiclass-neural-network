from network import MCNeuralNetwork, load_data

nodes = [784, 128, 64, 10]
epochs = 30
alpha = 0.01
batch_size = 128

test_features, test_labels, valid_features, valid_labels, train_features, train_labels = load_data()
net = MCNeuralNetwork(nodes, test_features, test_labels, valid_features, valid_labels, train_features, train_labels)

print("-" * 84)
print("Select an Option:\n1. Train Network\n2. Change Hyperparameters\n3. Test Network\n4. Reload and Shuffle Dataset\n5. Exit")
choice = int(input("Enter your choice: "))
print("-" * 84)

while choice != 5:
    if choice == 1:
        net.train(epochs, alpha, batch_size)

    elif choice == 2:
        nodes = [784]

        length = int(input("Enter the number of layers: "))

        while length <= 1:
            print("Number of layers must be greater than 1")
            length = int(input("Enter the number of layers: "))
            
        for i in range(length - 1):
            nodes.append(int(input(f"Enter the number of nodes in hidden layer {i + 1}: ")))

        nodes.append(10)

        epochs = int(input("Enter the number of epochs: "))
        alpha = float(input("Enter the learning rate: "))
        batch_size = int(input("Enter the batch size: "))

        net = MCNeuralNetwork(nodes, test_features, test_labels, valid_features, valid_labels, train_features, train_labels)

    elif choice == 3:
        net.test()

    elif choice == 4:
        test_features, test_labels, valid_features, valid_labels, train_features, train_labels = load_data()
        net.reload_data(test_features, test_labels, valid_features, valid_labels, train_features, train_labels)
        print("Dataset reloaded and shuffled successfully")
    
    print("-" * 84)
    print("Select an Option:\n1. Train Network\n2. Change Hyperparameters\n3. Test Network\n4. Reload and Shuffle Dataset\n5. Exit")
    choice = int(input("Enter your choice: "))
    print("-" * 84)

print("Program Exited")