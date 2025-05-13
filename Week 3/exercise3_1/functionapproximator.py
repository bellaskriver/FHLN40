import torch
import numpy as np
import matplotlib.pyplot as plt

class FunctionApproximator:
    
    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        self.model = self.buildModel(input_dimension, hidden_dimension, output_dimension)
        self.train_cost_history = None
        self.val_cost_history = None
    
    ### TASK 8: Change activation function ###
    def buildModel(self, input_dimension, hidden_dimension, output_dimension):
        nonlinearity = torch.nn.ReLU()
        modules = []
        modules.append(torch.nn.Linear(input_dimension, hidden_dimension[0]))
        modules.append(nonlinearity)
        for i in range(len(hidden_dimension)-1):
            modules.append(torch.nn.Linear(hidden_dimension[i], hidden_dimension[i+1]))
            modules.append(nonlinearity)
        
        modules.append(torch.nn.Linear(hidden_dimension[-1], output_dimension))
        
        model = torch.nn.Sequential(*modules)
        print(model)
        return model
        
    def predict(self,x):
        return self.model(x)
    
    def costFunction(self, f, f_pred):
        return torch.mean((f - f_pred)**2)
    
    def train(self, x, x_val, f, f_val, epochs, **kwargs):
        
        # Select optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        
        # Initialize history arrays
        self.train_cost_history = np.zeros(epochs)
        self.val_cost_history = np.zeros(epochs)
        
        # Training loop
        for epoch in range(epochs):
            f_pred = self.predict(x)
            cost = self.costFunction(f, f_pred)
            
            f_val_pred = self.predict(x_val)
            cost_val = self.costFunction(f_val, f_val_pred)
            
            self.train_cost_history[epoch] = cost
            self.val_cost_history[epoch] = cost_val
            
            # Set gradients to zero.
            self.optimizer.zero_grad()
            
            # Compute gradient (backwardpropagation)
            cost.backward(retain_graph=True)
            
            # Update parameters
            self.optimizer.step()
            
            if epoch % 100 == 0:
                # print("Cost function: " + cost.detach().numpy())
                print(f'Epoch: {epoch}, Cost: {cost.detach().numpy()}')

    def plotTrainingHistory(self, yscale='log'):
        """Plot the training history."""
        
        # Set up plot
        fig, ax = plt.subplots(figsize=(4,3))
        ax.set_title("Cost function history")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cost function $C$")
        plt.yscale(yscale)
        
        # Plot data
        ax.plot(self.train_cost_history, 'k-', label="training cost")
        ax.plot(self.val_cost_history, 'r--', label="validation cost")
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        plt.tight_layout()
        plt.savefig('cost-function-history.eps')
        plt.show()

        
        
        
        