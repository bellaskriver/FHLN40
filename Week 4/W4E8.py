import torch

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the nonconvex function
def nonconvex(x1, x2):
    fn = (1 - x1 / 2 + x1**5 + x2**3) * torch.exp(-(x1**2) - x2**2)
    return fn

# Generate input data in the range [-3, 3] x [-3, 3]
x1 = torch.linspace(-3, 3, 100)
x2 = torch.linspace(-3, 3, 100)
X1, X2 = torch.meshgrid(x1, x2)
inputs = torch.stack([X1.flatten(), X2.flatten()], dim=1)
targets = nonconvex(inputs[:, 0], inputs[:, 1])

# Define the neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # First hidden layer with 64 neurons
        self.fc2 = nn.Linear(64, 64)  # Second hidden layer with 64 neurons
        self.fc3 = nn.Linear(64, 1)  # Output layer
        self.relu = nn.ReLU()
        # self.ELU = nn.ELU() experiment with ELU activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the network, loss function, and optimizer
model = Network()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs.squeeze(), targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the loss curve
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# Plot the original function and the approximation
with torch.no_grad():
    predicted = model(inputs).reshape(100, 100)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Original function
ax1.plot_surface(X1.numpy(), X2.numpy(), targets.reshape(100, 100).numpy(), cmap='viridis')
ax1.set_title('Original Function')

# Approximated function
ax2.plot_surface(X1.numpy(), X2.numpy(), predicted.numpy(), cmap='viridis')
ax2.set_title('Approximated Function')

plt.show()