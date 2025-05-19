# Kommentar: 

# -*- coding: utf-8 -*-
import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax

# Function for u1
def u1(x1,x2):
        return torch.sin(x1) + torch.cos(2 * x2) + torch.cos(x1 * x2)

# Function for u2
def u2(x1,x2):
        return torch.cos(x1) + torch.sin(3 * x2) + torch.sin(x1 * x2)

# c o m p u t e d i s p l a c e m e n t g r a d i e n t s using a u t o g r a d
def gradient_calc(u1, u2, x1_mesh, x2_mesh):
    du1_dx1 = grad(u1, x1_mesh, torch.ones_like(u1), create_graph=True)[0]
    du1_dx2 = grad(u1, x2_mesh, torch.ones_like(u1), create_graph=True)[0]
    du2_dx1 = grad(u2, x1_mesh, torch.ones_like(u2), create_graph=True)[0]
    du2_dx2 = grad(u2, x2_mesh, torch.ones_like(u2), create_graph=True)[0]
    return du1_dx1, du1_dx2, du2_dx1, du2_dx2
# d e f i n e s t r a i n c o m p o n e n t s for plane s t r a i n c o n d i t i o n s

def strain_tensor(du1_dx1, du1_dx2, du2_dx1, du2_dx2):
    eps_11 = du1_dx1
    eps_22 = du2_dx2
    eps_12 = 0.5*(du1_dx2 + du2_dx1)
    return eps_11, eps_22, eps_12
# d e f i n e s t r e s s c o m p o n e n t s for plane s t r a i n c o n d i t i o n s
def stress_tensor(eps_11, eps_22, eps_12, G, v):
    sigma_11 = 2*G*(eps_11+v/(1-2*v)*(eps_11+eps_22))
    sigma_22 = 2*G*(eps_22+v/(1-2*v)*(eps_11+eps_22))
    sigma_33 = 2*G*v/(1-2*v)*(eps_11+eps_22)
    sigma_12 = 2*G*eps_12
    return sigma_11, sigma_22, sigma_33, sigma_12

# c o m p u t e d i v e r g e n c e of s t r e s s using a u t o g r a d
def divergence_of_stress(sigma_11, sigma_22, sigma_12, x1, x2):
    div1 = grad(sigma_11, x1, torch.ones_like(sigma_11), create_graph=True)[0] + grad(sigma_12, x2, torch.ones_like(sigma_12) ,create_graph=True)[0]
    div2 = grad(sigma_12, x1, torch.ones_like(sigma_12) ,create_graph=True)[0] + grad(sigma_22, x2, torch.ones_like(sigma_22), create_graph=True)[0]
    print("div1", div1)
    print("div2", div2)


    return div1, div2
# # main code
E = 1
#c o n s t a n t s
v = 0.25
G = E/(2*(1+v))
# set up i n p u t s and o u t p u t s a s w e l l as mesh
x1 = torch.linspace(0,2,100, requires_grad=True) # c h a n g e nbr of p o i n t s when code is r u n n i n g !!!
x2 = torch.linspace(0,1,50, requires_grad=True)
x1_mesh, x2_mesh = torch.meshgrid(x1,x2,indexing="ij")
u1 = u1_fn(x1_mesh, x2_mesh)
u2 = u2_fn(x1_mesh, x2_mesh)
# c o m p u t e r e q u i r e d r e s u l t s
gradient = gradient_calc(u1, u2, x1_mesh, x2_mesh)
strains = strain_tensor(gradient[0], gradient[1], gradient[2], gradient[3])
stresses = stress_tensor(strains[0], strains[1], strains[2], G, v)
div1, div2 = divergence_of_stress(stresses[0], stresses[1], stresses[3], x1_mesh, x2_mesh)
print("body␣forces:", -1*div1[0,0], -1*div2[0,0])

# # code for p l o t t i n g
x1_plot = x1_mesh.detach().numpy()
x2_plot = x2_mesh.detach().numpy()

# plot d i s p l a c e m e n t g r a d i e n t s
fig, axs = plt.subplots(2, 2, figsize=(12, 6),subplot_kw={"projection": "3d"})
for i, ax in enumerate(axs.flat):
    surf = ax.plot_surface(x1_plot, x2_plot, gradient[i].detach().numpy(), cmap="jet")
    ax.set_title(f"Displacement gradient {i+1}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("value␣of␣displacement␣gradient")
    fig.colorbar(surf, ax=ax)
# plot s t r a i n s
fig, axs = plt.subplots(1, 3, figsize=(12, 6),subplot_kw={"projection": "3d"})
for i, ax in enumerate(axs.flat):
    surf = ax.plot_surface(x1_plot, x2_plot, strains[i].detach().numpy(), cmap="jet")
    ax.set_title(f"Strain components {i+1}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("value␣of␣strains")
    fig.colorbar(surf, ax=ax)
# plot s t r e s s e s
# plot stresses
fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})
for i, ax in enumerate(axs.flat):
    surf = ax.plot_surface(
        x1_plot, x2_plot,
        stresses[i].detach().numpy(),
        cmap="jet"
    )
    ax.set_title(f"Stress {i+1}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("value of stresses")
    fig.colorbar(surf, ax=ax)

# show all four at once
plt.tight_layout()
plt.show()
