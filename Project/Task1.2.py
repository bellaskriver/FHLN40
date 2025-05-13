# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Exact solution for u1
def u1(x):
    return torch.sin(x[:, 0:1]) + torch.cos(2 * x[:, 1:2]) + torch.cos(x[:, 0:1] * x[:, 1:2])

# Exact solution for u2
def u2(x):
    return torch.cos(x[:, 0:1]) + torch.sin(3 * x[:, 1:2]) + torch.sin(x[:, 0:1] * x[:, 1:2])

# Dataset generator
def make_dataset(n_pts):
    x1 = torch.rand(n_pts, 1) * 2.0  # Random values between 0 and 2
    x2 = torch.rand(n_pts, 1) * 1.0  # Random values between 0 and 1
    X = torch.cat([x1, x2], dim=1) # Combined x1 and x2 [n_pts, 2]
    y1 = u1(X) # Exact solution for u1 based on x1 and x2
    y2 = u2(X) # Exact solution for u2 based on x1 and x2
    Y = torch.cat([y1, y2], dim=1) # Combined y1 and y2 [n_pts, 2]
    return X, Y

# Neural network model
class Network(nn.Module):
    def __init__(self, hidden_dim=20):
        super(Network, self).__init__() # Initialize nn.Module

        # Define the neural network architecture
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim), # Input layer (2)
            nn.Tanh(), # Activation function
            nn.Linear(hidden_dim, hidden_dim), # Hidden layer (20)
            nn.Tanh(), # Activation function
            nn.Linear(hidden_dim, 2) # Output layer (2)
        )

    # Forward pass through the network
    def forward(self, x):
        return self.network(x)

# Training function
def train_model(model, loader, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam optimizer with learning rate lr
    loss_fn = nn.MSELoss() # Mean Squared Error loss function
    
    # Training loop
    for ep in range(1, epochs + 1):
        model.train() # Training mode
        total_loss = 0.0 # Initialize total loss

        # Iterate over batches in the DataLoader
        for xb, yb in loader:
            optimizer.zero_grad() # Zero gradients
            preds = model(xb) # Forward pass
            loss = loss_fn(preds, yb) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights
            total_loss += loss.item() * xb.size(0) # Accumulate loss

        # Print progress
        if ep == 1 or ep % 100 == 0:
            avg_loss = total_loss / len(loader.dataset) # Average loss
            print(f"Epoch {ep}/{epochs}: Loss = {avg_loss:.3e}")
    return model

# Plotting function
def plot_results(model, X, Y, n_train):
    model.eval()
    
    # Evaluation mode (validation set) FIX
    with torch.no_grad():
        X_val = X[n_train:]
        Y_val = Y[n_train:]
        Y_pred = model(X_val).cpu().numpy()
        Y_exact = Y_val.cpu().numpy()

    ### Scatter plot
    Ys = {'u1': (Y_exact[:, 0], Y_pred[:, 0]), 'u2': (Y_exact[:, 1], Y_pred[:, 1])} # Datapoints for u1 and u2
    fig = plt.figure(figsize=(12,8))
    
    for label, (y_e, y_p) in Ys.items():
        plt.scatter(y_e, y_p, s=5, alpha=0.5, label=label) # Plotting datapoints
    mn, mx = min(Y_exact.min(), Y_pred.min()), max(Y_exact.max(), Y_pred.max()) # Min and max values for diagonal line
    plt.plot([mn, mx], [mn, mx], 'k-') # Diagonal line
    plt.xlabel('Exact')
    plt.ylabel('Predicted')
    plt.title('Exact vs. Predicted')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Meshgrid
    x1 = torch.linspace(0, 2, 100)
    x2 = torch.linspace(0, 1, 50)
    X2, X1 = torch.meshgrid(x2, x1, indexing='ij') # Create meshgrid
    grid = torch.cat([X1.reshape(-1,1), X2.reshape(-1,1)], dim=1) # Combine x1 and x2
    X1_np, X2_np = X1.cpu().numpy(), X2.cpu().numpy() # Convert to numpy for plotting

    # Calculate exact and predicted solutions
    with torch.no_grad():
        U1_ex = u1(grid).reshape(50, 100).cpu().numpy() 
        U2_ex = u2(grid).reshape(50, 100).cpu().numpy()
        U_pred = model(grid).cpu().numpy()
        U1_pred = U_pred[:,0].reshape(50, 100)
        U2_pred = U_pred[:,1].reshape(50, 100)

    # Titles and data for plots
    fields = [('Exact $u_1$',     U1_ex), ('Exact $u_2$',     U2_ex), ('Predicted $u_1$', U1_pred), ('Predicted $u_2$', U2_pred)]

    ### 2D plots
    fig, axes = plt.subplots(2, 2, figsize=(12,8), constrained_layout=True)
    for ax, (title, Z) in zip(axes.flat, fields):
        cf = ax.contourf(X1_np, X2_np, Z, levels=50)
        fig.colorbar(cf, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
    plt.show()

    ### 3D plots
    fig = plt.figure(figsize=(12,8))
    for idx, (title, Z) in enumerate(fields):
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        surf = ax.plot_surface(X1_np, X2_np, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title(title)
        ax.set_xlabel('$x_1$') 
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(title.split()[0])
        fig.colorbar(surf, ax=ax, shrink=0.5)
    plt.tight_layout() 
    plt.show()

# Main function
def main():
    # Generate data
    X, Y = make_dataset(5000)
    n_train = int(0.8 * len(X)) # 80% for training, 20% for validation

    # Prepare DataLoaders
    train_ds = TensorDataset(X[:n_train], Y[:n_train])
    val_ds = TensorDataset(X[n_train:], Y[n_train:])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256) #FIX

    # Initialize and train the model
    model = Network(hidden_dim=20)
    model = train_model(model, train_loader, epochs=1000, lr=1e-3)

    # Evaluate and plot results on validation set
    plot_results(model, X, Y, n_train)

# Run the main function
if __name__ == '__main__':
    main()