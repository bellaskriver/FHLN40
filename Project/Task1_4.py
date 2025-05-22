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

# PINN for static equilibrium: -div(sigma(u)) = b, with Dirichlet BC.
# Cost function: C = MSE_b + MSE_f
class PINN:
    def __init__(self, E, nu):
        self.E = E # Young's modulus
        self.nu = nu # Poisson's ratio
        self.net = self.build_net(2, [20,20,20,20], 2) # Network architecture, 2 inumpyuts, 2 outputs, 4 hidden layers with 20 neurons each

    # Build the neural network
    def build_net(self, in_dim, hidden, out_dim):
        layers = []
        dims = [in_dim] + hidden + [out_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    # Forward pass
    def forward(self, X):
        return self.net(X)

    # Calculate divergence
    def divergence(self, X, U):
        # Displacement components
        u1c = U[:,0:1]
        u2c = U[:,1:2] 

        # Displacement gradients
        du1 = torch.autograd.grad(u1c, X, torch.ones_like(u1c), create_graph=True)[0]
        du2 = torch.autograd.grad(u2c, X, torch.ones_like(u2c), create_graph=True)[0]
        u1_x1 = du1[:,0:1] 
        u1_x2 = du1[:,1:2]
        u2_x1 = du2[:,0:1] 
        u2_x2 = du2[:,1:2]

        # Strains
        eps11 = u1_x1
        eps22 = u2_x2
        eps12 = 0.5*(u1_x2 + u2_x1)

        # Lamé constants
        lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        mu = self.E/(2*(1+self.nu))

        # Stresses
        s11 = lam*(eps11+eps22) + 2*mu*eps11
        s22 = lam*(eps11+eps22) + 2*mu*eps22
        s12 = 2*mu*eps12

        # Divergence
        ds11 = torch.autograd.grad(s11, X, torch.ones_like(s11), create_graph=True)[0]
        ds12 = torch.autograd.grad(s12, X, torch.ones_like(s12), create_graph=True)[0]
        ds22 = torch.autograd.grad(s22, X, torch.ones_like(s22), create_graph=True)[0]
        div1 = ds11[:,0:1] + ds12[:,1:2]
        div2 = ds12[:,0:1] + ds22[:,1:2]

        return div1, div2

    # Body forces solved analytically
    def body_force(self, X):
        X.requires_grad_(True)
        U_true = torch.cat([u1(X[:,0:1], X[:,1:2]), u2(X[:,0:1], X[:,1:2])], dim=1)
        d1, d2 = self.divergence(X, U_true)
        return -d1, -d2

    # Prepare for training
    def prepare_training(self, nx, ny, lr):
        # Generate meshgrid
        x1 = torch.linspace(0,2,nx)
        x2 = torch.linspace(0,1,ny)
        X1,X2 = torch.meshgrid(x1,x2, indexing='ij')
        X_int = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)
        X_int.requires_grad_(True)

        # Calculate analytical body forces
        b1, b2 = self.body_force(X_int)
        self.X_int, self.b1, self.b2 = X_int, b1, b2
        self.opt = torch.optim.Adam(self.net.parameters(), lr) # Optimizer
        self.hist = {'mse_b':[], 'mse_f':[], 'cost':[]} # History for loss

    # Train the model
    def train(self, X_bnd, U_bnd, epochs):
        Nf = self.X_int.shape[0]
        Nb = X_bnd.shape[0]
        for it in range(epochs):
            self.opt.zero_grad() # Reset gradients

            # Partial Differential Equation (PDE) loss
            U_pred = self.forward(self.X_int)
            d1, d2 = self.divergence(self.X_int, U_pred)
            mse_f = ((d1 + self.b1)**2 + (d2 + self.b2)**2).sum() / Nf

            # Boundary Condition loss
            Ub = self.forward(X_bnd)
            mse_b = ((Ub - U_bnd)**2).sum() / Nb
            cost = mse_f + mse_b

            # Backpropagation
            cost.backward(retain_graph=True)
            self.opt.step()
            self.hist['mse_b'].append(mse_b.item())
            self.hist['mse_f'].append(mse_f.item())
            self.hist['cost'].append(cost.item())
            if it % 100 == 0 or it == epochs-1:
                print(f"Iteration: [{it}/{epochs-1}]  MSE_f={mse_f:.3e},  MSE_b={mse_b:.3e},  Cost={cost:.3e}")

        # Final backward without retain
        self.opt.zero_grad()
        mse_f = ((self.divergence(self.X_int, self.forward(self.X_int))[0] + self.b1)**2 + (self.divergence(self.X_int, self.forward(self.X_int))[1] + self.b2)**2).sum() / Nf
        mse_b = ((self.forward(X_bnd) - U_bnd)**2).sum() / Nb
        (mse_f + mse_b).backward()

    # Plotting function
    def plot(self, nx, ny):
        # Plot loss history
        fig=plt.figure(figsize=(12,8))
        plt.semilogy(self.hist['mse_b'], label='MSE_b')
        plt.semilogy(self.hist['mse_f'], label='MSE_f')
        plt.semilogy(self.hist['cost'], label='Cost')
        plt.legend() 
        plt.xlabel('Iteration') 
        plt.ylabel('Loss') 
        plt.show()

        # Create meshgrid for plotting
        x1 = torch.linspace(0,2,nx)
        x2 = torch.linspace(0,1,ny)
        X1,X2 = torch.meshgrid(x1,x2, indexing='ij')
        X = torch.stack([X1.reshape(-1), X2.reshape(-1)],1)
        U = self.forward(X).detach().numpy()
        U1 = U[:,0].reshape(nx,ny) 
        U2 = U[:,1].reshape(nx,ny)
        U1e = u1(X1,X2).numpy() 
        U2e = u2(X1,X2).numpy()

        # Plot 3D solution
        fig=plt.figure(figsize=(12,8))
        cmap = 'jet'
        for idx,(exact,pred,lab) in enumerate([(U1e,U1,'u1'),(U2e,U2,'u2')]):
            ax=fig.add_subplot(2,2,2*idx+1,projection='3d')
            ax.plot_surface(X1.numpy(), X2.numpy(), exact, cmap='jet')
            ax.set_title(f'Exact {lab}')
            ax=fig.add_subplot(2,2,2*idx+2,projection='3d')
            ax.plot_surface(X1.numpy(), X2.numpy(), pred, cmap='jet')
            ax.set_title(f'Predicted {lab}')
        plt.tight_layout(); plt.show()

        # Calculate residuals (Should be close to zero!)
        X.requires_grad_(True)
        U = self.forward(X)
        d1, d2 = self.divergence(X, U)
        b1, b2 = self.body_force(X)
        r1 = (d1 + b1).detach().numpy().reshape(nx, ny) # Residual of u1
        r2 = (d2 + b2).detach().numpy().reshape(nx, ny) # Residual of u2
        print('Max residual u1:', np.max(np.abs(r1))) # Largest residual positive or negative of u1
        print('Max residual u2:', np.max(np.abs(r2))) # Largest residual positive or negative of u2
        print('Mean residual u1:', np.mean(r1)) # Mean residual of u1
        print('Mean residual u2:', np.mean(r2)) # Mean residual of u2

        # Plot 2D residuals
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        im0 = axes[0].contourf(X1.numpy(), X2.numpy(), r1, cmap='jet')
        plt.colorbar(im0, ax=axes[0])
        axes[0].set_title('Residual 1')
        im1 = axes[1].contourf(X1.numpy(), X2.numpy(), r2, cmap='jet')
        plt.colorbar(im1, ax=axes[1])
        axes[1].set_title('Residual 2')
        plt.show()

# Main function
def main():
    E, nu = 1.0, 0.25 # Material parameters
    pinn = PINN(E, nu) # Initialize PINN

    # Extract Dirichlet BC
    n=10 # Number of points on each edge of the boundary
    edges = {
        'Left Boundary:': (torch.zeros(n), torch.linspace(0,1,n)),
        'Right Boundary:': (2*torch.ones(n), torch.linspace(0,1,n)),
        'Bottom Boundary:':(torch.linspace(0,2,n), torch.zeros(n)),
        'Top Boundary:': (torch.linspace(0,2,n), torch.ones(n))}
    Xb=[]

    # Print boundary values
    for name,(xs,ys) in edges.items():
        print(name)
        for x,y in zip(xs,ys):
            print(f"Point ({x:.2f},{y:.2f}): u1={u1(x.unsqueeze(0),y.unsqueeze(0))[0]:.3f}, u2={u2(x.unsqueeze(0),y.unsqueeze(0))[0]:.3f}")
        Xb.append(torch.stack([xs,ys],1))
    X_bnd = torch.cat(Xb,0)
    U_bnd = torch.cat([u1(X_bnd[:,0:1], X_bnd[:,1:2]), u2(X_bnd[:,0:1], X_bnd[:,1:2])],1)

    # Train the PINN
    pinn.prepare_training(nx=20, ny=20, lr=1e-3) # nx, ny: number of points in x1 and x2 directions, lr: learning rate
    pinn.train(X_bnd, U_bnd, epochs=3000) # epochs: number of training iterations

    # Plot training history and solution
    pinn.plot(nx=20,ny=20) # nx, ny: number of points in x1 and x2 directions

if __name__ == '__main__':
    main()