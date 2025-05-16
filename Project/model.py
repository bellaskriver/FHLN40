# P h y s i c s I n f o r m e d N e u r a l import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class PINN:
    "A␣class␣used␣to␣define␣the␣physics␣informed␣model␣in␣task␣1.4␣"
    def __init__(self, E, v, uB, xB, b1, b2):
        self.E = E
        self.v = v
        self.uB = uB
        self.xB = xB
        self.b1 = b1
        self.b2 = b2
        self.model = self.buildModel(2, [20,20,20,20], 2) # ( inputs , [# n e u r o n s per h i d d e n layer ] , o u t p u t s )
        self.differential_equation_cost_history = None
        self.boundary_condition_cost_history = None
        self.total_cost_history = None
        self.optimizer = None
        self.loss = None
        self.u_res = None

    def buildModel(self, input_dim, hidden_dims, output_dim):
        """ Build an MLP with given dimensions. """
        torch.manual_seed(2)
        layers = []
        # input → first hidden
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.Tanh())
        # remaining hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.Tanh())
        # finally the output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        return nn.Sequential(*layers)


    def getDisplacements(self,x):
        """ get d i s p l a c e m e n t s """
        u = self.model(x)
        u.requires_grad_(True)
        return u
    
    def getDivergence(self, x1, x2, u_pred):
        u1 = u_pred[:,0]
        u2 = u_pred[:,1]
        # c o m p u t e d i s p l a c e m e n t g r a d i e n t s
        eps1_x1 = torch.autograd.grad(u1, x1, torch.ones_like(u1), create_graph=True)[0]
        eps1_x2 = torch.autograd.grad(u1, x2, torch.ones_like(u1), create_graph=True)[0]
        eps2_x1 = torch.autograd.grad(u2, x1, torch.ones_like(u2), create_graph=True)[0]
        eps2_x2 = torch.autograd.grad(u2, x2, torch.ones_like(u2), create_graph=True)[0]
        # c o m p u t e s t r a i n c o m p o n e n t s
        eps1 = eps1_x1
        eps2 = eps2_x2
        gam12 = eps1_x2 + eps2_x1
        # c o m p u t e the n e e d e d s t r e s s c o m p o n e n t s
        sig1 = self.E/(1+self.v) * (eps1 + self.v/(1-2*self.v)*(eps1+eps2))
        sig2 = self.E/(1+self.v) * (eps2 + self.v/(1-2*self.v)*(eps1+eps2))
        tau12 = 0.5*self.E/(1+self.v) * gam12
        # c o m p u t e the n e e d e d d e r i v a t i v e s of the s t r e s s e s
        sig1_x1 = torch.autograd.grad(sig1, x1, torch.ones_like(sig1), create_graph=True)[0]
        sig2_x2 = torch.autograd.grad(sig2, x2, torch.ones_like(sig2), create_graph=True)[0]
        tau12_x1 = torch.autograd.grad(tau12, x1, torch.ones_like(tau12), create_graph=True)[0]
        tau12_x2 = torch.autograd.grad(tau12, x2, torch.ones_like(tau12), create_graph=True)[0]
        # c o m p u t e the d i v e r g e n c e 
        div1 = sig1_x1 + tau12_x2
        div2 = tau12_x1 + sig2_x2
        return div1, div2

    # of s t r e s s
    def costFunction(self, x1, x2, u_pred, uB, xB, b1, b2):
        """ c o m p u t e the cost f u n c t i o n """
        # c o m p u t e the d i f f e r e n t i a l e q u a t i o n cost
        div1, div2 = self.getDivergence(x1, x2, u_pred)
        b1 = b1*torch.ones_like(div1)
        b2 = b2*torch.ones_like(div2)
        cost_diff = torch.sum((div1 + b1)**2) + torch.sum((div2 + b2)**2)
        # c o m p u t e the b o u n d a r y cost
        upred_bound = self.getDisplacements(xB)
        cost_bound = torch.sum((upred_bound-uB)**2)
        return cost_diff, cost_bound

    def closure(self):
        """ c l o s u r e f u n c t i o n for the o p t i m i z e r """
        self.optimizer.zero_grad()
        u_pred = self.getDisplacements(self.x) # f o r w a r d p r o p a g a t i o n
        differential_equation_cost, boundary_condition_cost = self.costFunction(self.x1_mesh, self.x2_mesh, u_pred, self.uB, self.xB, self.b1, self.b2)
        loss = differential_equation_cost + boundary_condition_cost # c a l c u l a t e total cost
        loss.backward(retain_graph=True) # b a c k w a r d p r o p a g a t i o n
        return loss


    def train(self, samples_x1, samples_x2, epochs, **kwargs):
        """ train the model """
        x1 = torch.linspace(0,2,samples_x1, requires_grad=True)
        x2 = torch.linspace(0,1,samples_x2, requires_grad=True)
        x1_mesh, x2_mesh = torch.meshgrid(x1,x2,indexing="ij")
        self.x1_mesh = x1_mesh
        self.x2_mesh = x2_mesh
        x = torch.stack((x1_mesh, x2_mesh), dim=2).view(-1,2)
        self.x = x
        # Set o p t i m i z e r
        self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        # I n i t i a l i z e h i s t o r y a r r a y s
        self.differential_equation_cost_history = np.zeros(epochs)
        self.boundary_condition_cost_history = np.zeros(epochs)
        self.total_cost_history = np.zeros(epochs)
        self.u_res = 0

        # T r a i n i n g loop
        for i in range(epochs):
            # P r e d i c t d i s p l a c e m e n t s
            u_pred = self.getDisplacements(x)
            # Cost f u n c t i o n c a l c u l a t i o n
            differential_equation_cost, boundary_condition_cost = self.costFunction(x1_mesh, x2_mesh, u_pred, self.uB, self.xB, self.b1, self.b2)
            # Total cost
            total_cost = differential_equation_cost + boundary_condition_cost
            # Add e n e r g y v a l u e s to h i s t o r y
            self.differential_equation_cost_history[i] += differential_equation_cost
            self.boundary_condition_cost_history[i] += boundary_condition_cost
            self.total_cost_history[i] += total_cost
            self.u_res = u_pred
            # Print t r a i n i n g state
            self.printTrainingState(i, epochs)
            # U p d a t e p a r a m e t e r s
            self.optimizer.step(self.closure)
        self.x = None

    def printTrainingState(self, epoch, epochs, print_every=100):
        """ Print the cost v a l u e s of the c u r r e n t epoch in a t r a i n i n g loop . """
        if epoch == 0 or epoch == (epochs - 1) or epoch % print_every == 0 or print_every == "all":
            # P r e p a r e s t r i n g
            string = "Epoch:␣{}/{}\t\tDifferential␣equation␣cost␣=␣{:2f}\t\tBoundary␣condition␣cost␣=␣{:2f}\t\tTotal␣cost␣=␣{:2f}"
            # F o r m a t s t r i n g and print
            print(string.format(epoch, epochs - 1, self.differential_equation_cost_history[epoch],
            self.boundary_condition_cost_history[epoch], self.total_cost_history[epoch]))

    def plotTrainingHistory(self, yscale="log"):
        """ Plot the t r a i n i n g h i s t o r y . """
        # Set up plot
        fig, ax = plt.subplots()
        ax.set_title("Cost␣function␣history")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cost␣function␣$C$ ")
        plt.yscale(yscale)
        # Plot data
        ax.plot(self.differential_equation_cost_history, label="Differential␣equation␣cost")
        ax.plot(self.boundary_condition_cost_history, label="Boundary␣condition␣cost")
        ax.plot(self.total_cost_history, label="Total␣cost")
        ax.legend()


    def plotDisplacements(self, u1_analytic, u2_analytic):
        """ Plot d i s p l a c e m e n t s . """
        x1_plot = self.x1_mesh.detach().numpy()
        x2_plot = self.x2_mesh.detach().numpy()
        # Set up plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})
        # plot r e s p e c t i v e s u r f a c e g r a p h s
        surf1 = ax1.plot_surface(x1_plot, x2_plot, u1_analytic, cmap="jet")
        ax1.set_title("Displacements␣for␣manufactured␣solution␣u1(x1,x2)")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_zlabel("u1")
        fig.colorbar(surf1, ax=ax1)
        surf2 = ax2.plot_surface(x1_plot, x2_plot, self.u_res[:,0].view(-1,2).reshape(x1_plot.shape).detach().numpy(), cmap="jet")
        ax2.set_title("Displacements␣for␣trained␣solution␣u1(x1,x2)")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        ax2.set_zlabel("u1")
        fig.colorbar(surf2, ax=ax2)
        surf3 = ax3.plot_surface(x1_plot, x2_plot, u2_analytic, cmap="jet")
        ax3.set_title("Displacements␣for␣manufactured␣solution␣u2(x1,x2)")
        ax3.set_xlabel("x1")
        ax3.set_ylabel("x2")
        ax3.set_zlabel("u2")
        fig.colorbar(surf3, ax=ax3)
        surf4 = ax4.plot_surface(x1_plot, x2_plot, self.u_res[:,1].view(-1,2).reshape(x1_plot.shape).detach().numpy(), cmap="jet")
        ax4.set_title("Displacements␣for␣trained␣solution␣u2(x1,x2)")
        ax4.set_xlabel("x1")
        ax4.set_ylabel("x2")
        ax4.set_zlabel("u2")
        fig.colorbar(surf4, ax=ax4)