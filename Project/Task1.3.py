# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
from pyDOE import lhs
import torch

# d e f i n e the n e u r a l n e t w o r k
class Network(torch.nn.Module):
    def __init__(self, inputdim, hidden1dim, hidden2dim, outputdim):
        super().__init__()
        self.input   = torch.nn.Linear(inputdim,   hidden1dim)
        self.hidden1 = torch.nn.Linear(hidden1dim, hidden2dim)
        # ← fix second hidden: input is hidden2dim
        self.hidden2 = torch.nn.Linear(hidden2dim, hidden2dim)
        self.output  = torch.nn.Linear(hidden2dim, outputdim)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = self.output(x)    # ← actually capture the final linear layer’s output
        return x              # ← now is size (batch_size, outputdim)

# d e f i n e the o u t p u t functions , i . e . 
def u1_fn(x1,x2):
        u1 = torch.sin(x1) + torch.cos(2 * x2) + torch.cos(x1 * x2)

        return u1
# the m a n u f a c t u r e d d i s p l a c e m e n t s
def u2_fn(x1,x2):
        u2 = torch.cos(x1) + torch.sin(3 * x2) + torch.sin(x1 * x2)
        return u2
    

# set up the input and o u t p u t data a s w e l l as mesh
x1 = torch.linspace(0,2,20)
x2 = torch.linspace(0,1,20)
x1_mesh, x2_mesh = torch.meshgrid(x1,x2,indexing="ij")
x = torch.stack((x1_mesh, x2_mesh), dim=2).view(-1,2)
u1 = u1_fn(x1_mesh, x2_mesh)
u2 = u2_fn(x1_mesh, x2_mesh)
y = torch.stack((u1, u2), dim=2).view(-1,2)
# set up v a l i d a t i o n set using lhs
val_set = torch.tensor(lhs(2,10), dtype=torch.float32)
x1_val_mesh, x2_val_mesh = torch.meshgrid(2*val_set[:,0], val_set[:,1], indexing="ij")
xval = torch.stack((x1_val_mesh, x2_val_mesh), dim=2).view(-1,2)
u1_val = u1_fn(x1_val_mesh, x2_val_mesh)
u2_val = u2_fn(x1_val_mesh, x2_val_mesh)
y_val = torch.stack((u1_val, u2_val), dim=2).view(-1,2)
# c r e a t e the n e t w o r k
torch.manual_seed(2)
inputdim = 2 # nbr of i n p u t s - x1 & x2
hidden1dim = 20 # nbr n e u r o n s in first h i d d e n layer
hidden2dim = 20 # nbr n e u r o n s in s e c o n d h i d d e n layer
outputdim = 2 # nbr of o u t p u t s - u1 & u2
model = Network(inputdim, hidden1dim, hidden2dim, outputdim)

# d e f i n e the o p t i m i z e r and cost f u n c t i o n
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
cost_fnc = torch.nn.MSELoss()

# train the n e t w o r k
n_epochs = 2000

# i n i t i a l i z e h i s t o r y a r r a y s
valcost_history = np.zeros(n_epochs)
traincost_history = np.zeros(n_epochs)
for epoch in range(n_epochs):
    optimizer.zero_grad()
    y_pred = model(x) # f o r w a r d p r o p a g a t i o n on t r a i n i n g set
    val_pred = model(xval) # f o r w a r d p r o p a g a t i o n on v a l i d a t i o n set
    cost = cost_fnc(y_pred, y) # c a l c u l a t e cost of t r a i n i n g set
    val_cost = cost_fnc(val_pred, y_val) # c a l c u l a t e cost of v a l i d a t i o n set
    cost.backward() # b a c k w a r d p r o p a g a t i o n on t r a i n i n g set
    optimizer.step() # u p d a t e p a r a m e t e r s
    if epoch%100==0:
        print(f"Epoch␣{epoch},␣Training␣Cost␣{cost.item():10.6e}␣Validation␣Cost␣{val_cost.item():10.6e}")

# i n s e r t v a l u e s in h i s t o r y a r r a y s
valcost_history[epoch] = val_cost.item()
traincost_history[epoch] = cost.item()
valcost_history[epoch] += cost
traincost_history[epoch] += val_cost
# code for p l o t t i n g
# plot the graph of u1 and u 1 _ p r e d
x1_plot = x1_mesh.detach().numpy()
x2_plot = x2_mesh.detach().numpy()
u1_analytical = y[:,0].reshape(x1_plot.shape)
u1_pred = y_pred[:,0].reshape(x1_plot.shape).detach().numpy()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})
# m a n u f a c t u r e d s o l u t i o n plot
surf1 = ax1.plot_surface(x1_plot, x2_plot, u1_analytical, cmap="jet")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("u2")
ax1.set_title("graph␣of␣manufactured␣solution␣u1")
fig.colorbar(surf1, ax=ax1)

# a p p r o x i m a t e d s o l u t i o n plot
surf2 = ax2.plot_surface(x1_plot, x2_plot, u1_pred, cmap="jet")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_zlabel("u1")
ax2.set_title("graph␣of␣trained␣solution␣for␣u1")
fig.colorbar(surf2, ax=ax2)

# plot the graph of u2 and u 2 _ p r e d
u2_analytical = y[:,1].reshape(x1_plot.shape)
u2_pred = y_pred[:,1].reshape(x1_plot.shape).detach().numpy()

# m a n u f a c t u r e d s o l u t i o n plot
surf3 = ax3.plot_surface(x1_plot, x2_plot, u2_analytical, cmap="jet")
ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.set_zlabel("u2")
ax3.set_title("graph␣of␣manufactured␣solution␣u2")
fig.colorbar(surf3, ax=ax3)

# a p p r o x i m a t e d s o l u t i o n plot
surf4 = ax4.plot_surface(x1_plot, x2_plot, u2_pred, cmap="jet")
ax4.set_xlabel("x1")
ax4.set_ylabel("x2")
ax4.set_zlabel("u2")
ax4.set_title("graph␣of␣trained␣solution␣for␣u2")
fig.colorbar(surf4, ax=ax4)

# plot t r a i n i n g and v a l i d a t i o n cost fig, ax = plt.subplots()
# plot training and validation cost
fig2, ax = plt.subplots()        # ← new 2D axes
ax.set_title("Cost function history")
ax.set_xlabel("Epochs")
ax.set_ylabel("Cost function $C$")
ax.plot(valcost_history, label="Validation cost")
ax.plot(traincost_history, label="Training cost")
ax.legend()
ax.set_yscale("log")
plt.show()
