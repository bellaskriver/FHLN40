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
    X = torch.cat([x1, x2], dim=1) 
    y1 = u1(X)
    y2 = u2(X)
    Y = torch.cat([y1, y2], dim=1)
    return X, Y

# Neural network model
class Network(nn.Module):
    def __init__(self, hidden_dim=20):
        super(Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.network(x)

# Training function with history recording
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_losses = []
    val_losses = []

    for ep in range(1, epochs + 1):
        # Training step
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # Validation step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = loss_fn(preds, yb)
                total_val_loss += loss.item() * xb.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if ep == 1 or ep % 100 == 0:
            print(f"Epoch {ep}/{epochs}: Train Loss = {avg_train_loss:.3e}, Val Loss = {avg_val_loss:.3e}")

    return model, train_losses, val_losses

# Plotting loss history
def plot_loss_history(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Training loss')
    plt.plot(epochs, val_losses, label='Validation loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plotting function for results
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
        surf = ax.plot_surface(X1_np, X2_np, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')
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
    n_train = int(0.8 * len(X))

    # Prepare DataLoaders
    train_ds = TensorDataset(X[:n_train], Y[:n_train])
    val_ds = TensorDataset(X[n_train:], Y[n_train:])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    # Initialize and train the model
    model = Network(hidden_dim=20)
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=100, lr=1e-3)

    # Plot the loss history
    plot_loss_history(train_losses, val_losses)

    # Evaluate and plot results on validation set
    plot_results(model, X, Y, n_train)

if __name__ == '__main__':
    main()
