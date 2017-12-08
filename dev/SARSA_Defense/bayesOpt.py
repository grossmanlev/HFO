"""
Author: Patrick Hansen

BayesianOptimizer object runs bayesian optimization over an evaluation function
given a range of legal values for each hyperparameter.

Ex:
    bo = BayesianOptimizer(my_func,
                           [[arg1_min,arg1_max],[arg2_min,arg2_max],...],
                           maximize=True)
    bo.run(10)
    myParams = bo.getBestParams()
"""

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

import matplotlib.pyplot as plt

class BayesianOptimizer:
    def __init__(self, eval_function, bounds, maximize=False,
                 alpha=1e-5, epsilon=1e-7):
        self.eval_function = eval_function
        self.maximize = maximize
        self.alpha = alpha
        self.epsilon = epsilon

        self.bounds = np.array(bounds)
        self.nb_params = self.bounds.shape[0]

        self.xs = []
        self.ys = []

        self.model = gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(),
                                                 alpha=alpha,
                                                 n_restarts_optimizer=10,
                                                 normalize_y=True)

    # Calucates the expected improvement of parameter values x from model
    # Algorithm from https://arxiv.org/pdf/1705.10033.pdf
    def expectedImprovement(self, x):
        # Get predicted distrubution
        mu, sigma = self.model.predict(x.reshape(-1, self.nb_params),
                                       return_std=True)

        # Find optimal observed point
        if self.maximize:
            optimum = max(self.ys)
            k = 1
        else:
            optimum = min(self.ys)
            k = -1

        # Calculate expected improvement
        # Avoid division by zero
        if sigma:
            Z = k*(mu-optimum)/sigma
            res = -(k*(mu-optimum)*norm.cdf(Z) + sigma*norm.pdf(Z))
        else:
            res = 0.0

        return res

    # Returns the parameter values with the greatest expected improvement
    # Uses L-BFGS-B optimizer from scipy
    def getNextSample(self, nb_restarts=25):
        best_x = None
        best_v = float('inf')

        # Optimize function with many starting points to avoid local minima
        x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                               size=(nb_restarts, self.nb_params))

        for x in x0:
            # Minimize negative expected improvement
            res = minimize(fun=self.expectedImprovement,
                           x0=x.reshape(1, -1),
                           bounds=self.bounds,
                           method='L-BFGS-B')

            # Store best operating point
            if res.fun < best_v:
                best_v = res.fun
                best_x = res.x

        # Cannot sample same point
        # If x is within epsilon, then choose random
        while np.any(np.abs(best_x - np.array(self.xs)) <= self.epsilon):
            best_x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                       size=self.nb_params)

        return best_x

    # Populates data before fitting the model
    def presample(self, x0=None, nb_pre_samples=5):
        # Sample randomly if no initial points specified
        if x0 is None:
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                   size=(nb_pre_samples, self.nb_params))

        for x in x0:
            # Sample evaluation function
            y = self.eval_function(x)

            # Update data
            self.xs.append(list(x))
            self.ys.append(y)

    # Runs bayesian optimization
    def run(self, nb_iters):
        # Presample if needed (random settings)
        if not self.xs:
            self.presample()

        for _ in range(nb_iters):
            # Fit gaussian processes to observed data
            # Sklearn uses arrays
            self.model.fit(np.array(self.xs), np.array(self.ys))

            # Sample evaluation function
            x = self.getNextSample()
            y = self.eval_function(x)

            # Update data
            self.xs.append(list(x))
            self.ys.append(y)

    # Returns all parameter values and evaluated outputs
    def getData(self):
        return (self.xs, self.ys)

    # Returns best parameter values
    def getBestSample(self):
        if self.maximize:
            idx = self.ys.index(max(self.ys))
        else:
            idx = self.ys.index(min(self.ys))

        return (self.xs[idx], self.ys[idx])
