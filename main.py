from network import MCNeuralNetwork

nodes = [784, 128, 64, 10]
epochs = 30
alpha = 0.01
batch_size = 128

net = MCNeuralNetwork(nodes)

print("-" * 60)
print("Select an Option:\n1. Train Network\n2. Change Hyperparameters\n3. Test Network\n4. Exit")
choice = int(input("Enter your choice: "))
print("-" * 60)

while choice != 4:
    if choice == 1:
        net.train(epochs, alpha, batch_size=128)

    elif choice == 2:
        length = int(input("Enter the number of layers: "))
        for i in range(length - 1):
            nodes[i + 1] = int(input(f"Enter the number of nodes in hidden layer {i + 1}: "))

        epochs = int(input("Enter the number of epochs: "))
        alpha = float(input("Enter the learning rate: "))
        batch_size = int(input("Enter the batch size: "))

        net = MCNeuralNetwork(nodes)

    elif choice == 3:
        net.test()
    
    print("-" * 60)
    print("Select an Option:\n1. Train Network\n2. Change Hyperparameters\n3. Test Network\n4. Exit")
    choice = int(input("Enter your choice: "))
    print("-" * 60)