"""
Adaptation of the Expected Improvement acquisition function from GPFlowOpt to include known constraint
on the domain that should be explored. Note this is slightly different to (and more straightforward than)
the case where there is an unknown constraint.
"""


from gpflowopt.acquisition import Acquisition


from gpflow.model import Model
from gpflowopt.domain import ContinuousParameter
from gpflow.param import DataHolder
from gpflow import settings

import numpy as np
import tensorflow as tf

stability = settings.numerics.jitter_level



def heaviside(x):
    """
    Computes the heaviside (step) function in TensorFlow. Returns 1 if x >0, 0 if x<=0.
    Args:
        x:

    Returns:

    """
    return 0.5 * (1 + tf.math.sign(x))



class ConstrainedExpectedImprovement(Acquisition):
    """
    Constrained Expected Improvement acquisition function for single-objective global optimization.
    Introduced by (Mockus et al, 1975).

    Key reference:

    ::

       @article{Jones:1998,
            title={Efficient global optimization of expensive black-box functions},
            author={Jones, Donald R and Schonlau, Matthias and Welch, William J},
            journal={Journal of Global optimization},
            volume={13},
            number={4},
            pages={455--492},
            year={1998},
            publisher={Springer}
       }

    This acquisition function is the expectation of the improvement over the current best observation
    w.r.t. the predictive distribution. The definition is closely related to the :class:`.ProbabilityOfImprovement`,
    but adds a multiplication with the improvement w.r.t the current best observation to the integral.

    .. math::
       \\alpha(\\mathbf x_{\\star}) = \\int \\max(f_{\\min} - f_{\\star}, 0) \\, p( f_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} ) \\, d f_{\\star}
    """

    def __init__(self, model, constraint):
        """

        Args:
            model: GPflow model (single output) representing our belief of the objective
            constraint: A function g, which describes the known constraint on the domain. Effectively allows us to
                        explore a domain with a different shape to a hyper-rectangle. Note, we wish to find the value of x
                        that maximises the unknown function f, subject to the known constraint g(x) > 0.
        """

        super(ConstrainedExpectedImprovement, self).__init__(model)
        self.fmin = DataHolder(np.zeros(1))
        self.constraint = constraint
        self._setup()

    def _setup(self):
        super(ConstrainedExpectedImprovement, self)._setup()
        # Obtain the lowest posterior mean for the previous - feasible - evaluations
        feasible_samples = self.data[0][self.highest_parent.feasible_data_index(), :]
        samples_mean, _ = self.models[0].predict_f(feasible_samples)
        self.fmin.set_data(np.min(samples_mean, axis=0))

    def build_acquisition(self, Xcand):
        # Obtain predictive distributions for candidates
        candidate_mean, candidate_var = self.models[0].build_predict(Xcand)
        candidate_var = tf.maximum(candidate_var, stability)

        delta = self.constraint(Xcand)
        pof = heaviside(delta)

        # Compute EI
        normal = tf.contrib.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
        t1 = (self.fmin - candidate_mean) * normal.cdf(self.fmin)
        t2 = candidate_var * normal.prob(self.fmin)
        return pof * tf.add(t1, t2, name=self.__class__.__name__)