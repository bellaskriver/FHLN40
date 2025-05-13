import torch
import math
import matplotlib.pyplot as plt
from functionapproximator import FunctionApproximator

# Plot settings
import matplotlib
from matplotlib import rc
matplotlib.rcParams["figure.dpi"] = 80
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

# Set seed for the Random Number Generator (RNG)
torch.manual_seed(0)

func = lambda x: torch.sin(2*math.pi*x)

lb = -1.0   # lower bound
ub = 1.0    # upper bound

n_samples = 40  # number of samples
noise = 0.1     # random noise added to the training and validation data

# initialize a uniform random distribution
m = torch.distributions.uniform.Uniform(torch.tensor([lb]), torch.tensor([ub]))

# create training data and corresponding labels
x_train = m.sample(torch.tensor([n_samples]))
y_train = func(x_train) + noise * m.sample(torch.tensor([n_samples]))

# create validation data and corresponding labels
x_validate = m.sample(torch.tensor([n_samples]))
y_validate = func(x_validate) + noise * m.sample(torch.tensor([n_samples]))

### TASK 1 & 2: Change network architecture ###
# network architecture
input_dimension = 1
hidden_neurons = [50,50]  # e.g. [20, 20] for 2 hidden layers with 20 neurons each
output_dimension = 1

# creat model according to defined network architecture
Model = FunctionApproximator(input_dimension,
                             hidden_neurons,
                             output_dimension)

### TASK 5: Change learning rate ###
# optimization parameters
epochs = 10000
learning_rate = 0.001  # learning rate

### TASK 6 & 7: Apply L2-regularization ###
# regularization parameter
lamb = 1e-3

# train the model feeding the training and the validation data
Model.train(x_train,
            x_validate,
            y_train,
            y_validate,
            epochs,
            lr=learning_rate,
            weight_decay=lamb
            )

### TASK 4: Plot cost function history ###
# plot training history
# Model.plotTrainingHistory()

# generate a test set from the analytic sine function
x_test = torch.linspace(lb,ub,100).view(-1,1)
y_pred = Model.predict(x_test)
y_analytic = func(x_test)

# plot the analytic solution, the network prediciton and the training examples
# .detach() simply returns the tensor without the corresponding gradients
plt.figure(figsize=(4, 3))
plt.scatter(x_train.detach(), y_train.detach(), marker='x', c='k', s=10, label='training data', zorder=2)
plt.plot(x_test.detach(), y_analytic.detach(), '-', color='silver', linewidth=2, label='sin(x)', zorder=1)
plt.plot(x_test.detach(), y_pred.detach(), 'r--', linewidth=2, label='prediction', zorder=3)
plt.title('Prediction of the sine function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper center', bbox_to_anchor=(0.43, -0.2), ncol=3)
plt.tight_layout()
plt.savefig('prediction.eps')
