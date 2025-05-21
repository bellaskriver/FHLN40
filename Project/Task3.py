# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as ag
import numpy as np

class PINN(nn.Module):
    """Class for the PINN model."""
    def __init__(self, hidden_layers=[50,50,50,50]):
        super().__init__()
        layers = []
        in_dim = 2
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers += [nn.Linear(in_dim, 2)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        # x: [N,2], returns u: [N,2]
        return self.net(x)

#-------------------------------------------------------
# 2) helper: compute stresses and equilibrium residuals
#-------------------------------------------------------
def elastic_residual(x, model, λ=0.4, μ=0.4):
    """
    x: [N,2] with requires_grad=True
    returns res: [N,2] = [∂σ11/∂x1 + ∂σ12/∂x2,
                          ∂σ12/∂x1 + ∂σ22/∂x2]
    """
    u = model(x)                      # [N,2]
    u1 = u[:,0:1];  u2 = u[:,1:2]

    # first derivatives
    grads = ag.grad(u, x,
                    grad_outputs=torch.ones_like(u),
                    create_graph=True)
    # grads is a tuple (d[u1,u2]/dx) same shape as u, but batched
    # better: compute each separately:
    du1 = ag.grad(u1, x, torch.ones_like(u1), create_graph=True)[0]
    du2 = ag.grad(u2, x, torch.ones_like(u2), create_graph=True)[0]
    u1_x = du1[:,0:1]; u1_y = du1[:,1:2]
    u2_x = du2[:,0:1]; u2_y = du2[:,1:2]

    # strains
    ε11 = u1_x
    ε22 = u2_y
    ε12 = 0.5*(u1_y + u2_x)

    # stresses (plane strain)
    σ11 = λ*(ε11+ε22) + 2*μ*ε11
    σ22 = λ*(ε11+ε22) + 2*μ*ε22
    σ12 = 2*μ*ε12

    # equilibrium: divergence of σ
    σ11_x = ag.grad(σ11, x, torch.ones_like(σ11), create_graph=True)[0][:,0:1]
    σ12_y = ag.grad(σ12, x, torch.ones_like(σ12), create_graph=True)[0][:,1:2]
    σ12_x = ag.grad(σ12, x, torch.ones_like(σ12), create_graph=True)[0][:,0:1]
    σ22_y = ag.grad(σ22, x, torch.ones_like(σ22), create_graph=True)[0][:,1:2]

    res1 = σ11_x + σ12_y
    res2 = σ12_x + σ22_y
    return res1, res2

#-------------------------------------------------------
# 3) boundary conditions sampling
#-------------------------------------------------------
def sample_interior(n):
    x1 = torch.rand(n,1)*2.0
    x2 = torch.rand(n,1)*1.0
    x  = torch.cat([x1,x2], dim=1)
    x.requires_grad_(True)
    return x

def sample_bottom(n):
    x1 = torch.rand(n,1)*2.0
    x2 = torch.zeros_like(x1)
    return torch.cat([x1,x2],1)

def sample_sides(n):
    # n on left, n on right
    x2 = torch.rand(n,1)*1.0
    left  = torch.cat([torch.zeros_like(x2),   x2],1)
    right = torch.cat([2*torch.ones_like(x2), x2],1)
    return torch.cat([left,right],0)

def sample_top(n):
    x1 = torch.rand(n,1)*2.0
    x2 = torch.ones_like(x1)*1.0
    return torch.cat([x1,x2],1)

#-------------------------------------------------------
# 4) training loop
#-------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = PINN().to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

λ, μ = 0.4, 0.4
σ0  = 0.1

for it in range(1_000_01):
    # 1) PDE residual loss
    x_int = sample_interior(2000).to(device)
    r1, r2 = elastic_residual(x_int, model, λ, μ)
    loss_pde = (r1**2 + r2**2).mean()

    # 2) Dirichlet BC at bottom
    xb = sample_bottom(500).to(device)
    ub = model(xb)
    loss_dir = (ub**2).mean()

    # 3) traction BC on top
    xt = sample_top(500).to(device)
    xt.requires_grad_(True)
    # compute stresses at xt
    res1, res2 = elastic_residual(xt, model, λ, μ)  
    # but we want stress itself
    u = model(xt)
    du1 = ag.grad(u[:,0:1], xt, torch.ones_like(u[:,0:1]), create_graph=True)[0]
    du2 = ag.grad(u[:,1:2], xt, torch.ones_like(u[:,1:2]), create_graph=True)[0]
    u1_x,u1_y = du1[:,0:1], du1[:,1:2]
    u2_x,u2_y = du2[:,0:1], du2[:,1:2]
    ε11,ε22,ε12 = u1_x, u2_y, 0.5*(u1_y+u2_x)
    σ11 = λ*(ε11+ε22)+2*μ*ε11
    σ22 = λ*(ε11+ε22)+2*μ*ε22
    σ12 = 2*μ*ε12

    # traction = σ·n, n=(0,1) => t1=σ12, t2=σ22
    t1 = σ12; t2 = σ22

    # build target traction
    tgt_t1 = torch.zeros_like(t1)
    # t2 = 0.1 if x1 in [2/3,4/3]
    cond = ((xt[:,0:1]>=2/3)&(xt[:,0:1]<=4/3)).float()
    tgt_t2 = σ0*cond

    loss_top = ((t1-tgt_t1)**2 + (t2-tgt_t2)**2).mean()

    # 4) traction‐free on sides: σ11=σ12=0 at x1=0,2
    xs = sample_sides(250).to(device)
    xs.requires_grad_(True)
    u_s = model(xs)
    du1s = ag.grad(u_s[:,0:1], xs, torch.ones_like(u_s[:,0:1]), create_graph=True)[0]
    du2s = ag.grad(u_s[:,1:2], xs, torch.ones_like(u_s[:,1:2]), create_graph=True)[0]
    s11 = λ*(du1s[:,0:1]+du2s[:,1:2]) + 2*μ*du1s[:,0:1]
    s12 = μ*(du1s[:,1:2]+du2s[:,0:1])
    loss_sides = (s11**2 + s12**2).mean()

    # total loss
    loss = loss_pde + 10*loss_dir + 10*loss_top + 10*loss_sides

    opt.zero_grad()
    loss.backward()
    opt.step()

    if it%1000==0:
        print(f"it {it:6d}  loss_pde {loss_pde.item():.3e}  loss_dir {loss_dir.item():.3e}"
              +f"  loss_top {loss_top.item():.3e}  loss_sides {loss_sides.item():.3e}")

# after training you can evaluate u on a grid and plot:
import matplotlib.pyplot as plt
nx, ny = 100,50
X = np.linspace(0,2,nx)
Y = np.linspace(0,1,ny)
xx,yy = np.meshgrid(X,Y)
xy = torch.tensor(np.stack([xx.reshape(-1), yy.reshape(-1)],axis=1),dtype=torch.float32).to(device)
with torch.no_grad():
    uu = model(xy).cpu().numpy().reshape(-1,2)
U1 = uu[:,0].reshape(ny,nx)
plt.contourf(xx,yy,U1,50); plt.colorbar()
plt.title("u1 displacement field"); plt.show()