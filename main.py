import torch
from torch import nn
import random
import matplotlib.pyplot as plt

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 55)
        self.fc2 = nn.Linear(55, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Initialize the network
net = Net()

# Print the network architecture
print(net)

# Count the number of parameters
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"The network has {num_params} parameters.")

# Define the boxers
boxer1 = {"name": "Boxer 1", "health": 100}
boxer2 = {"name": "Boxer 2", "health": 100}

# Define the number of rounds
rounds = 10

# Prepare a list to store the health of each boxer after each round
health_history = {"Boxer 1": [100], "Boxer 2": [100]}

# Simulate the match
for round in range(rounds):
    # Boxer 1's attack
    attack = random.randint(1, 20)
    boxer2["health"] -= attack

    # Check if Boxer 2 has been knocked out
    if boxer2["health"] <= 0:
        health_history["Boxer 2"].append(0)
        break
    else:
        health_history["Boxer 2"].append(boxer2["health"])

    # Boxer 2's attack
    attack = random.randint(1, 20)
    boxer1["health"] -= attack

    # Check if Boxer 1 has been knocked out
    if boxer1["health"] <= 0:
        health_history["Boxer 1"].append(0)
        break
    else:
        health_history["Boxer 1"].append(boxer1["health"])

# Plot the health history
plt.figure(figsize=(10, 6))
plt.plot(health_history["Boxer 1"], label="Boxer 1")
plt.plot(health_history["Boxer 2"], label="Boxer 2")
plt.xlabel("Round")
plt.ylabel("Health")
plt.title("Health of Boxers Over Time")
plt.legend()
plt.grid(True)
plt.show()
