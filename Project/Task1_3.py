# -*- coding: utf-8 -*-
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt

# Function for u1
def u1(x1,x2):
    return torch.sin(x1) + torch.cos(2 * x2) + torch.cos(x1 * x2)

# Function for u2
def u2(x1,x2):
    return torch.cos(x1) + torch.sin(3 * x2) + torch.sin(x1 * x2)

# Gradient
def gradient(u1, u2, x1_mesh, x2_mesh):
    du1_dx1 = grad(u1, x1_mesh, torch.ones_like(u1), create_graph=True)[0]
    du1_dx2 = grad(u1, x2_mesh, torch.ones_like(u1), create_graph=True)[0]
    du2_dx1 = grad(u2, x1_mesh, torch.ones_like(u2), create_graph=True)[0]
    du2_dx2 = grad(u2, x2_mesh, torch.ones_like(u2), create_graph=True)[0]
    return du1_dx1, du1_dx2, du2_dx1, du2_dx2

# Strain tensor
def strain(du1_dx1, du1_dx2, du2_dx1, du2_dx2):
    eps_11 = du1_dx1
    eps_22 = du2_dx2
    eps_12 = 0.5*(du1_dx2 + du2_dx1)
    return eps_11, eps_22, eps_12

# Stress tensor
def stress(eps_11, eps_22, eps_12, G, v):
    sigma_11 = 2*G*(eps_11 + v/(1-2*v)*(eps_11+eps_22))
    sigma_22 = 2*G*(eps_22 + v/(1-2*v)*(eps_11+eps_22))
    sigma_33 = 2*G*(v/(1-2*v)*(eps_11+eps_22))
    sigma_12 = 2*G*eps_12
    return sigma_11, sigma_22, sigma_33, sigma_12

# Divergence of stress
def divergence(sigma_11, sigma_22, sigma_12, x1, x2):
    div1 = grad(sigma_11, x1, torch.ones_like(sigma_11), create_graph=True)[0] + grad(sigma_12, x2, torch.ones_like(sigma_12), create_graph=True)[0]
    div2 = grad(sigma_12, x1, torch.ones_like(sigma_12), create_graph=True)[0] + grad(sigma_22, x2, torch.ones_like(sigma_22), create_graph=True)[0]
    return div1, div2

# Plotting function
def plot(x1_mesh, x2_mesh, gradients, strains, stresses, divs):
    x1_plot = x1_mesh.detach().numpy()
    x2_plot = x2_mesh.detach().numpy()

    # Plotting displacement gradients
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={"projection": "3d"})
    for i, ax_ in enumerate(axs.flat):
        surf = ax_.plot_surface(x1_plot, x2_plot, gradients[i].detach().numpy(), cmap="jet")
        ax_.set_title(f"Displacement gradient {i+1}")
        ax_.set_xlabel("x1"); ax_.set_ylabel("x2"); ax_.set_zlabel("value")
        fig.colorbar(surf, ax=ax_)
    plt.tight_layout()
    plt.show()

    # Plotting strain tensors
    fig, axs = plt.subplots(1, 3, figsize=(12, 8), subplot_kw={"projection": "3d"})
    for i, ax_ in enumerate(axs.flat):
        surf = ax_.plot_surface(x1_plot, x2_plot, strains[i].detach().numpy(), cmap="jet")
        ax_.set_title(f"Strain tensor {i+1}")
        ax_.set_xlabel("x1"); ax_.set_ylabel("x2"); ax_.set_zlabel("value")
        fig.colorbar(surf, ax=ax_)
    plt.tight_layout()
    plt.show()

    # Plotting stress tensors
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={"projection": "3d"})
    for i, ax_ in enumerate(axs.flat):
        surf = ax_.plot_surface(x1_plot, x2_plot, stresses[i].detach().numpy(), cmap="jet")
        ax_.set_title(f"Stress tensor {i+1}")
        ax_.set_xlabel("x1"); ax_.set_ylabel("x2"); ax_.set_zlabel("value")
        fig.colorbar(surf, ax=ax_)
    plt.tight_layout()
    plt.show()

    # Plotting divergence of stress
    fig, axs = plt.subplots(1, 2, figsize=(12, 8), subplot_kw={"projection": "3d"})
    for i, ax_ in enumerate(axs.flat):
        surf = ax_.plot_surface(x1_plot, x2_plot, divs[i].detach().numpy(), cmap="jet")
        ax_.set_title(f"Divergence component {i+1}")
        ax_.set_xlabel("x1"); ax_.set_ylabel("x2"); ax_.set_zlabel("value")
        fig.colorbar(surf, ax=ax_)
    plt.tight_layout()
    plt.show()

# Main function
def main(): 
    # Material parameters
    E = 1.0
    v = 0.25
    G = E/(2*(1+v))

    # Set up mesh
    x1 = torch.linspace(0,2,100, requires_grad=True)
    x2 = torch.linspace(0,1,50,  requires_grad=True)
    x1_mesh, x2_mesh = torch.meshgrid(x1, x2, indexing="ij")

    # Displacements, gradients, strains and stresses
    u1_val = u1(x1_mesh, x2_mesh)
    u2_val = u2(x1_mesh, x2_mesh)
    gradients = gradient(u1_val, u2_val, x1_mesh, x2_mesh)
    strains = strain(*gradients)
    stresses = stress(*strains, G, v)

    # Body forces
    div1, div2 = divergence(stresses[0], stresses[1], stresses[3], x1_mesh, x2_mesh)
    divs = [div1, div2]
    print("Body forces at (0,0):", -div1[0,0].item(), -div2[0,0].item())

    # Plot results
    plot(x1_mesh, x2_mesh, gradients, strains, stresses, divs)

# Run the main function
if __name__ == "__main__":
    main()
