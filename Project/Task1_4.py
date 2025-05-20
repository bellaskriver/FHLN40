# Fix kvar

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Function for u1
def u1(x1, x2):
    return torch.sin(x1) + torch.cos(2 * x2) + torch.cos(x1 * x2)

# Function for u2
def u2(x1, x2):
    return torch.cos(x1) + torch.sin(3 * x2) + torch.sin(x1 * x2)

# PINN
class PINN:
    def __init__(self, E, v, uB, xB, b1, b2):
        self.E, self.v = E, v # Material properties
        self.uB, self.xB = uB, xB 
        self.b1, self.b2 = b1, b2

        # Build model
        self.model = self.build_model(2, [20, 20, 20, 20], 2)

        # Placeholders for mesh and history
        self.d_eq_cost_hist = None
        self.bnd_cost_hist = None
        self.total_cost_hist = None
        self.optimizer = None
        self.u_pred_mesh = None

    def build_model(self, in_dim, hidden_dims, out_dim):
        torch.manual_seed(2)
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.clone().requires_grad_(True)
        u = self.model(x)
        return u

    def divergence(self, x1, x2, u_pred):
        u1, u2 = u_pred[:,0], u_pred[:,1]
        # displacement gradients
        grads = lambda f, v: torch.autograd.grad(f, v, torch.ones_like(f), create_graph=True)[0]
        u1_x1, u1_x2 = grads(u1, x1), grads(u1, x2)
        u2_x1, u2_x2 = grads(u2, x1), grads(u2, x2)

        # strains
        eps1, eps2 = u1_x1, u2_x2
        gamma12     = u1_x2 + u2_x1

        # stresses (plane strain)
        factor = self.E / (1 + self.v)
        sum_eps = eps1 + eps2
        sig1  = factor * ( eps1 + self.v/(1-2*self.v)*sum_eps )
        sig2  = factor * ( eps2 + self.v/(1-2*self.v)*sum_eps )
        tau12 = 0.5 * factor * gamma12

        # divergence of stress
        sig1_x1 = grads(sig1,  x1)
        sig2_x2 = grads(sig2,  x2)
        t12_x1  = grads(tau12, x1)
        t12_x2  = grads(tau12, x2)

        div1 = sig1_x1 + t12_x2
        div2 = t12_x1  + sig2_x2
        return div1, div2

    def loss_function(self, x1_mesh, x2_mesh):
        # forward pass on mesh
        X_flat = torch.stack([x1_mesh.reshape(-1), x2_mesh.reshape(-1)], dim=1)
        u_pred = self.forward(X_flat)
        div1, div2 = self.divergence(x1_mesh, x2_mesh, u_pred)

        # PDE residual cost
        b1_t = self.b1 * torch.ones_like(div1)
        b2_t = self.b2 * torch.ones_like(div2)
        cost_pde = ((div1 + b1_t)**2 + (div2 + b2_t)**2).sum()

        # boundary cost
        uB_pred = self.forward(self.xB)
        cost_bnd = ((uB_pred - self.uB)**2).sum()

        return cost_pde, cost_bnd

    def train(self, nx, ny, epochs, lr=1e-3):
        # create mesh grid for PDE
        x1 = torch.linspace(0, 2, nx, requires_grad=True)
        x2 = torch.linspace(0, 1, ny, requires_grad=True)
        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
        self.x1_mesh, self.x2_mesh = X1, X2

        # optimizer & history buffers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.d_eq_cost_hist = np.zeros(epochs)
        self.bnd_cost_hist  = np.zeros(epochs)
        self.total_cost_hist= np.zeros(epochs)

        # training loop
        for i in range(epochs):
            self.optimizer.zero_grad()
            cost_pde, cost_bnd = self.loss_function(X1, X2)
            loss = cost_pde + cost_bnd
            loss.backward()
            self.optimizer.step()

            # record history
            self.d_eq_cost_hist[i] = cost_pde.item()
            self.bnd_cost_hist[i]  = cost_bnd.item()
            self.total_cost_hist[i]= loss.item()

            # simple print every 100 iters
            if i % 100 == 0 or i == epochs-1:
                print(f"Epoch {i}/{epochs-1}  PDE Cost = {cost_pde:.3e}  Bnd Cost = {cost_bnd:.3e}")

    def plot_history(self, yscale="log"):
        plt.figure(figsize=(8,5))
        plt.plot(self.d_eq_cost_hist, label="PDE residual")
        plt.plot(self.bnd_cost_hist,  label="Boundary loss")
        plt.plot(self.total_cost_hist,label="Total loss")
        plt.yscale(yscale)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_displacements(self, u1_m, u2_m):
        X1_np = self.x1_mesh.detach().numpy()
        X2_np = self.x2_mesh.detach().numpy()
        U_pred = self.forward(torch.stack([self.x1_mesh.reshape(-1),
                                          self.x2_mesh.reshape(-1)],1))
        U1_pred = U_pred[:,0].reshape(X1_np.shape).detach().numpy()
        U2_pred = U_pred[:,1].reshape(X1_np.shape).detach().numpy()

        titles = [
            ("Exact $u_1$", u1_m),
            ("Predicted $u_1$", U1_pred),
            ("Exact $u_2$", u2_m),
            ("Predicted $u_2$", U2_pred),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12,8), constrained_layout=True,
                                 subplot_kw={"projection":"3d"})
        for ax, (title, Z) in zip(axes.flat, titles):
            surf = ax.plot_surface(X1_np, X2_np, Z, cmap="jet", edgecolor="none")
            ax.set_title(title)
            ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel(title.split()[1])
            fig.colorbar(surf, ax=ax, shrink=0.6)

# Main function
def main():
    # Manufactured boundary
    print("\n--- Manufactured boundary values ---")
    corners = torch.tensor([[0,0],
                            [2,0],
                            [2,1],
                            [0,1]], dtype=torch.float32)
    x1b, x2b = corners[:,0], corners[:,1]
    u1b = u1(x1b, x2b)
    u2b = u2(x1b, x2b)
    for (x1i, x2i, u1i, u2i) in zip(x1b, x2b, u1b, u2b):
        print(f"x1={x1i:.2f}, x2={x2i:.2f} → u1={u1i:.3f}, u2={u2i:.3f}")

    # Material & body‐force parameters
    E, v = 1.0, 0.25
    b1, b2 = -0.184, -0.104

    # Boundary coords/values for PINN
    xB = corners                    # [4×2]
    uB = torch.stack([u1b, u2b], 1) # [4×2]

    # Instantiate & train PINN
    pinn = PINN(E, v, uB, xB, b1, b2)
    pinn.train(nx=20, ny=20, epochs=1000, lr=1e-3)

    # Plot training history and solutions
    pinn.plot_history()
    x1m, x2m = torch.meshgrid(torch.linspace(0,2,20),
                              torch.linspace(0,1,20), indexing="ij")
    pinn.plot_displacements(u1(x1m, x2m),
                            u2(x1m, x2m))
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
