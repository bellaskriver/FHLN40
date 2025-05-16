# d e v e l o p a PINN which s o l v e s the s t a t i c e q u i l i b r i u m # main p r o g r a m for task 4 , rest of code in P I N N _ m o d e l
from model import PINN
import matplotlib.pyplot as plt
import torch
#e q u a t i o n
# d e f i n e a n a l y t i c a l s o l u t i o n s

def u1_fn(x1,x2):
        u1 = torch.sin(x1) + torch.cos(2 * x2) + torch.cos(x1 * x2)

        return u1
# the m a n u f a c t u r e d d i s p l a c e m e n t s
def u2_fn(x1,x2):
        u2 = torch.cos(x1) + torch.sin(3 * x2) + torch.sin(x1 * x2)
        return u2


# --- after you’ve defined u1_fn, u2_fn and (optionally) trained your model ---

# number of points per edge
n_edge = 20

# bottom edge: x2 = 0, x1 from 0→2
x1_bot = torch.linspace(0, 2, n_edge)
x2_bot = torch.zeros(n_edge)

# top edge:    x2 = 1, x1 from 0→2
x1_top = torch.linspace(0, 2, n_edge)
x2_top = torch.ones(n_edge)

# left edge:   x1 = 0, x2 from 0→1
x1_left = torch.zeros(n_edge)
x2_left = torch.linspace(0, 1, n_edge)

# right edge:  x1 = 2, x2 from 0→1
x1_right = 2 * torch.ones(n_edge)
x2_right = torch.linspace(0, 1, n_edge)

# pack them up
edges = {
    "bottom (x2=0)" : (x1_bot,  x2_bot),
    "top    (x2=1)" : (x1_top,  x2_top),
    "left   (x1=0)" : (x1_left, x2_left),
    "right  (x1=2)" : (x1_right,x2_right),
}

# Evaluate & print
for name, (x1_e, x2_e) in edges.items():
    u1_e = u1_fn(x1_e, x2_e)
    u2_e = u2_fn(x1_e, x2_e)
    print(f"\n{name}:")
    print("   x1     x2      u1       u2")
    for xi, yi, u1i, u2i in zip(x1_e.tolist(), x2_e.tolist(),
                                u1_e.tolist(),   u2_e.tolist()):
        print(f" {xi:5.2f}  {yi:5.2f}  {u1i:8.4f}  {u2i:8.4f}")


# given data
E = 1 # Y o u n g s m o d u l u s
v = 0.25 # p o i s s o n s ratio
b1 = -0.184 # body force c a l c u l a t e d in first task - u1
b2 = -0.104 # body force c a l c u l a t e d in first task - u2
xB = torch.tensor([[0,0], [2,0], [2,1], [0,1]], dtype = torch.float32) # b o u n d a r y c o n d i t i o n c o o r d i n a t e s
uB = torch.tensor([[0,0], [0.22, 0.42], [0.6, 1.1], [0.12, 0.22]], dtype = torch.float32) # b o u n d a r y c o n d i t i o n v a l u e s
# g e n e r a t e model
PINNmodel = PINN(E, v, uB, xB, b1, b2)
# train model
samples_x1 = 20
samples_x2 = 20
epochs = 5000
PINNmodel.train(samples_x1, samples_x2, epochs, lr=5e-4)
# Plot t r a i n i n g h i s t o r y
PINNmodel.plotTrainingHistory()
# Plot d i s p l a c e m e n t s
x1_mesh, x2_mesh = torch.meshgrid(torch.linspace(0,2,samples_x1),torch.linspace(0,1,samples_x2),indexing="ij")
u1 = u1_fn(x1_mesh, x2_mesh)
u2 = u2_fn(x1_mesh, x2_mesh)
PINNmodel.plotDisplacements(u1,u2)
plt.show()