import torch
import matplotlib.pyplot as plt

# definition of a grid
# define a 2-dimensional grid from 0 to 1 in both dimensions with 100 grid points each
x =
y =

x.requires_grad = True
y.requires_grad = True

# sampling of a function u
u = lambda x, y : torch.cat((2*x+3*y, 4*x+y),0)
U = u(x.unsqueeze(0), y.unsqueeze(0))

# estimation of the derivatives
# compute the derivatives of U with respect to x and y

# plot
fig, ax = plt.subplots()
plt.gca().set_aspect('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('$\\frac{du_{2}}{dx}$')
# plot the derivatives in a contourplot
cp = 
fig.colorbar(cp)
fig.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.set_title('$\\frac{du_{2}}{dx}$')
# plot the derivatives in a surfaceplot
cp=
fig.colorbar(cp, pad=0.2)
fig.tight_layout()
plt.show()