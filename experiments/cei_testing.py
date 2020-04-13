import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import pdb


from gpflowopt.bo import BayesianOptimizer
from gpflowopt.design import LatinHyperCube

from gpflowopt.optim import SciPyOptimizer, StagedOptimizer, MCOptimizer
from gpflowopt.domain import ContinuousParameter
from gpflow.model import Model
from gpflow.param import DataHolder
from gpflow import settings

from constrained_ei import ConstrainedExpectedImprovement
from experiment_utils import plotfx, test_function3d, volume_constraint3d

pdb.set_trace()

stability = settings.numerics.jitter_level





V = 10.0

#Specify the hyperrectangle domain
domain = ContinuousParameter('x1', 0, V) + \
         ContinuousParameter('x2', 0, V) + \
         ContinuousParameter('x3', 0, V)

#Specify the function and constraint
function = test_function3d
constraint = volume_constraint3d



# Use standard Gaussian process Regression
lhd = LatinHyperCube(10, domain)
X = lhd.generate()
print(X.shape)
print(X)
Y = function(X)
print(Y)
pdb.set_trace()

#Plot to look at the function we will be optimising
plotfx(function, constraint)
pdb.set_trace()
#Specify the model
model = gpflow.gpr.GPR(X, Y, gpflow.kernels.Matern52(2, ARD=True))
model.kern.lengthscales.transform = gpflow.transforms.Log1pe(1e-3)


# Now create the Bayesian Optimizer
alpha = ConstrainedExpectedImprovement(model, constraint)

acquisition_opt = StagedOptimizer([MCOptimizer(domain, 200),
                                   SciPyOptimizer(domain)])

optimizer = BayesianOptimizer(domain, alpha, optimizer=acquisition_opt, verbose=True)

# Run the Bayesian optimization
r = optimizer.optimize(function, n_iter=10)
print(r)