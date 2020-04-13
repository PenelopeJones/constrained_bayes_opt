import numpy as np
import matplotlib.pyplot as plt

import gpflowopt
from gpflowopt.domain import ContinuousParameter

import pdb

def volume_constraint3d(X):
    """
    The constraint on our system is that the sum of Volumes A, B and C must be <= Total Volume V.
    Thus the constraint is g(X) = V_T - V_A - V_B - V_C >=0.
    Args:
        X:

    Returns:

    """
    V = 10.0
    ret = V - X[:, 0] - X[:, 1] - X[:, 2]
    return np.reshape(ret, (-1, 1))

def test_function3d(x):
    """
    An example of an 'unknown' 3d function that we are trying to optimise. Use for testing the algorithm works.
    Args:
        x:

    Returns:

    """
    x = np.atleast_3d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    d = -1.0
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    ret = a * (x2 - b * x1 ** 2 + c * x1 + d*x3 - r) ** 2 + s * (1 - t) * np.cos(x1 + x2**2) + s

    return np.reshape(ret, (-1, 1))

def plotfx(function, constraint):
    """
    Visualise the test function and the domain which can be explored.
    Args:
        function: a function of the same number of variables as the constraint
        constraint: a function.

    Returns:

    """
    V = 10.0
    filename = 'testing'
    domain = ContinuousParameter('x1', 0, V) + \
             ContinuousParameter('x2', 0, V) + \

    X = gpflowopt.design.FactorialDesign(101, domain).generate()

    Zo = function(X)
    Zc = constraint(X)

    mask = Zc<=0
    Zc[mask] = np.nan
    Zc[np.logical_not(mask)] = 1
    Z = Zo * Zc
    shape = (101, 101)
    pdb.set_trace()
    f = plt.figure(figsize=(7, 7))
    axes = f.add_subplot(1, 1, 1)
    axes.contourf(X[0:101**2,0].reshape(shape), X[0:101**2,1].reshape(shape), Z[0:101**2].reshape(shape))
    f.savefig(filename + '.png')
    axes.set_xlabel('x1')
    axes.set_ylabel('x2')
    axes.set_xlim([domain.lower[0], domain.upper[0]])
    axes.set_ylim([domain.lower[1], domain.upper[1]])



    return