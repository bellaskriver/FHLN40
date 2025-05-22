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

# PINN for static equilibrium with Dirichlet and Neumann BC.
# Cost function: C = MSE_b + MSE_f + MSE_n
class PINN:
    def __init__(self, E, nu):
        self.E = E # Young's modulus
        self.nu = nu # Poisson's ratio
        self.net = self.build_net(2, [20, 20, 20], 2) # Network architecture

    # Build the neural network
    def build_net(self, in_dim, hidden, out_dim):
        layers = []
        dims = [in_dim] + hidden + [out_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            # Tanh activation on all hidden layers
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

        return div1, div2, s11, s22, s12

    # Calculate analytical body force from u1 and u2 (exact)
    def body_force(self, X):
        X.requires_grad_(True)
        U_true = torch.cat([u1(X[:,0:1],X[:,1:2]), u2(X[:,0:1],X[:,1:2])], dim=1) # Evaluate exact solution
        d1, d2, *_ = self.divergence(X, U_true) # Calculate divergence of exact solution
        return -d1, -d2

    # Prepare training
    def prepare_training(self, nx, ny, lr):
        # Generate meshgrid
        x1 = torch.linspace(0,2,nx)
        x2 = torch.linspace(0,1,ny)
        X1,X2 = torch.meshgrid(x1, x2, indexing='ij')
        X_int = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)
        X_int.requires_grad_(True)

        # Calculate analytic body forces inside domain
        b1, b2 = self.body_force(X_int)
        self.X_int = X_int
        self.b1 = b1.detach()
        self.b2 = b2.detach()

        self.opt = torch.optim.Adam(self.net.parameters(), lr) # Optimizer
        self.hist = {'mse_b':[], 'mse_f':[], 'mse_n':[], 'cost':[]} # History for loss

    # Train the model
    def train(self, X_bnd, U_bnd, X_neu, n_neu, t_neu, epochs):
        for it in range(epochs):
            self.opt.zero_grad()

            # PDE loss: -div(sigma(u_pred)) = b
            U_int = self.forward(self.X_int)
            d1, d2, *_ = self.divergence(self.X_int, U_int)
            mse_f = ((d1 + self.b1)**2 + (d2 + self.b2)**2).mean()

            # Dirichlet loss: match u_pred = u_exact on bottom edge
            U_b = self.forward(X_bnd)
            mse_b = ((U_b - U_bnd)**2).mean()

            # Neumann loss: match traction t_pred = sigma(u_pred)·n on free/loaded edges
            U_n = self.forward(X_neu)
            _, _, s11, s22, s12 = self.divergence(X_neu, U_n)
            n1, n2 = n_neu[:,0:1], n_neu[:,1:2]

            # Traction prediction
            t1p = s11*n1 + s12*n2
            t2p = s12*n1 + s22*n2

            # Target traction (zero on free edges, (0,sigma0) on loaded top segment)
            t1t, t2t = t_neu[:,0:1], t_neu[:,1:2]
            mse_n = ((t1p - t1t)**2 + (t2p - t2t)**2).mean()
            
            # Total cost
            cost = mse_f + mse_b + mse_n # Rescaling the loss
            cost.backward()
            self.opt.step()

            # Log loss values
            self.hist['mse_f'].append(mse_f.item())
            self.hist['mse_b'].append(mse_b.item())
            self.hist['mse_n'].append(mse_n.item())
            self.hist['cost'].append(cost.item())

            # print progress
            if it % 100 == 0 or it == epochs-1:
                print(f"Iteration: [{it}/{epochs}]  MSE_f={mse_f:.2e},  MSE_b={mse_b:.2e},  MSE_n={mse_n:.2e},  Cost={cost:.2e}")

    def plot(self, nx, ny):
        # Plot loss history
        plt.figure(figsize=(12,8))
        plt.semilogy(self.hist['mse_f'], label='MSE_f')
        plt.semilogy(self.hist['mse_b'], label='MSE_b')
        plt.semilogy(self.hist['mse_n'], label='MSE_n')
        plt.semilogy(self.hist['cost'], label='Cost')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout() 
        plt.show()

        # Create meshgrid for plotting
        x1 = torch.linspace(0,2,nx)
        x2 = torch.linspace(0,1,ny)
        X1,X2 = torch.meshgrid(x1, x2, indexing='ij')
        X = torch.stack([X1.reshape(-1), X2.reshape(-1)],1)
        U = self.forward(X).detach().numpy()
        U1p = U[:,0].reshape(nx,ny)
        U2p = U[:,1].reshape(nx,ny)
        U1e = u1(X1,X2).numpy()
        U2e = u2(X1,X2).numpy()

        # Plot 3D solution
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(2,2,1, projection='3d')
        ax.plot_surface(X1.numpy(), X2.numpy(), U1e, cmap='jet'); ax.set_title('Exact $u_1$')
        ax = fig.add_subplot(2,2,2, projection='3d')
        ax.plot_surface(X1.numpy(), X2.numpy(), U1p, cmap='jet'); ax.set_title('Predicted $u_1$')
        ax = fig.add_subplot(2,2,3, projection='3d')
        ax.plot_surface(X1.numpy(), X2.numpy(), U2e, cmap='jet'); ax.set_title('Exact $u_2$')
        ax = fig.add_subplot(2,2,4, projection='3d')
        ax.plot_surface(X1.numpy(), X2.numpy(), U2p, cmap='jet'); ax.set_title('Predicted $u_2$')
        plt.tight_layout() 
        plt.show()

        # Calculate residuals (Should be close to zero!)
        X.requires_grad_(True)
        Uc = self.forward(X)
        d1, d2, *_ = self.divergence(X, Uc)
        b1, b2 = self.body_force(X)
        R1 = (d1 + b1).detach().numpy().reshape(nx, ny)
        R2 = (d2 + b2).detach().numpy().reshape(nx, ny)
        print('Max residual u1:', np.max(np.abs(R1))) # Largest residual positive or negative of u1
        print('Max residual u2:', np.max(np.abs(R2))) # Largest residual positive or negative of u2
        print('Mean residual u1:', np.mean(R1)) # Mean residual of u1
        print('Mean residual u2:', np.mean(R2)) # Mean residual of u2

        # Plot residuals
        fig, axes = plt.subplots(1,2, figsize=(12,8))
        im = axes[0].contourf(X1.numpy(), X2.numpy(), R1, levels=20, cmap='jet')
        fig.colorbar(im, ax=axes[0]); axes[0].set_title('Residual $r_1$')
        im = axes[1].contourf(X1.numpy(), X2.numpy(), R2, levels=20, cmap='jet')
        fig.colorbar(im, ax=axes[1]); axes[1].set_title('Residual $r_2$')
        plt.tight_layout() 
        plt.show()

# Main function
def main():
    E, nu = 1.0, 0.25 # Material parameters
    sigma0 = 0.1 # Prescribed traction magnitude
    pinn = PINN(E, nu)

    # Dirichlet BC: bottom edge u = 0 and u = exact u on [0,2]
    nD = 100 # Number of points on bottom edge
    xb = torch.linspace(0,2,nD) 
    yb = torch.zeros(nD)
    Xb = torch.stack([xb, yb],1)
    Ub = torch.cat([u1(xb.unsqueeze(1), yb.unsqueeze(1)), u2(xb.unsqueeze(1), yb.unsqueeze(1))],1)

    # Neumann BC pieces (left, right, loaded top, free tops)
    nN = 100 # Number of points on each edge of the boundary

    # Left edge x=0: traction-free => t = (0,0)
    yl = torch.linspace(0,1,nN)
    XL = torch.stack([torch.zeros(nN), yl],1)
    nL = torch.tensor([[-1.0,0.0]]).repeat(nN,1)
    tL = torch.zeros(nN,2)

    # Right edge x=2: traction-free => t=(0,0)
    yr = torch.linspace(0,1,nN)
    XR = torch.stack([2*torch.ones(nN), yr],1)
    nR = torch.tensor([[1.0,0.0]]).repeat(nN,1)
    tR = torch.zeros(nN,2)

    # Loaded top edge [2/3,4/3]x{1}: loaded with vertical traction sigma0
    xt = torch.linspace(2/3,4/3,nN)
    XTL = torch.stack([xt, torch.ones(nN)],1)
    nTL = torch.tensor([[0.0,1.0]]).repeat(nN,1)
    tTL = torch.stack([torch.zeros(nN), sigma0*torch.ones(nN)],1)

    # Free-top left segment: traction-free
    xf1 = torch.linspace(0,2/3,nN)
    XTFL = torch.stack([xf1, torch.ones(nN)],1)
    nTFL = torch.tensor([[0.0,1.0]]).repeat(nN,1)
    tTFL = torch.zeros(nN,2)

    # Free-top right segment: traction-free
    xf2 = torch.linspace(4/3,2,nN)
    XTFR = torch.stack([xf2, torch.ones(nN)],1)
    nTFR = torch.tensor([[0.0,1.0]]).repeat(nN,1)
    tTFR = torch.zeros(nN,2)

    # Combine all Neumann BC pieces
    Xn = torch.cat([XL, XR, XTL, XTFL, XTFR], dim=0)
    nnorm = torch.cat([nL, nR, nTL, nTFL, nTFR], dim=0)
    tn = torch.cat([tL, tR, tTL, tTFL, tTFR], dim=0)
    Xn.requires_grad_(True)

    # Train the PINN
    pinn.prepare_training(nx=100, ny=100, lr=1e-4) # nx, ny: number of points in x1 and x2 directions, lr: learning rate
    pinn.train(Xb, Ub, Xn, nnorm, tn, epochs=5000) # epochs: number of training iterations

    # Plot training history and solution
    pinn.plot(nx=100, ny=100) # nx, ny: number of points in x1 and x2 directions

if __name__=='__main__':
    main()