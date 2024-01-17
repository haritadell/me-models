import os
import itertools
import numpy as np
from scipy.stats import dirichlet, multivariate_normal, multivariate_t
import scipy.spatial.distance as distance
import jax.numpy as jnp
import jax
from jax import vmap, value_and_grad, jit, pmap
from jax.config import config
from sklearn.linear_model import LinearRegression
import scipy.odr as odr
from jax.example_libraries import optimizers
import math
import matplotlib.pyplot as plt
from utils import k_jax

# NPL class
class npl():
    """This class contains functions to perform NPL inference for any of the models in models.py.
    The user supplies parameters:
        data: Data set
        B: number of bootstrap iterations
        m: number of points sampled from P_\theta at each optim. iteration
        c: concentration parameter of Dirichlet Process
        T: truncation limit for DP
        p: number of unknown parameters
        lx: lengthscale of gaussian kernel for X: set -1 to get the median heuristic
        ly: lengthscale of gaussian kernel for Y: set -1 to get the median heuristic
        seed: random seed for Dirichlet weights
        prior: prior value for measurement error std
    """

    def __init__(self, data, B, m, c, T, seed, lx, ly, prior):
        self.B = int(B)
        self.m = m
        self.T = int(T)  # truncation limit for DP sample approximation
        self.c = c  # DP concentration parameter
        self.data = data
        self.n = len(self.data[:,0]) # number of observations
        self.seed = seed
        self.p = 3 # number of unknown parameters
        self.lx = lx
        self.ly = ly
        if self.lx == -1:
            self.lx = np.sqrt((1/2)*np.median(distance.cdist(self.data[:,0].reshape((self.n,1)),self.data[:,0].reshape((self.n,1)),'sqeuclidean')))
        if self.ly == -1:
            self.ly = np.sqrt((1/2)*np.median(distance.cdist(self.data[:,1].reshape((self.n,1)),self.data[:,1].reshape((self.n,1)),'sqeuclidean')))
        self.prior = prior
        self.w_groups, self.group_sizes = np.unique(self.data[:,0], return_counts=True)
        self.num_groups = len(self.w_groups)
        

    def draw_single_sample(self, weights, x_tilde, key):
        """ Draws a single sample from the nonparametric posterior specified via
        the model and the data X"""

        theta_j = self.minimise_MMD(self.data, weights, x_tilde, key)

        return theta_j

    def loss(self, rng, theta, D, xsample):
        """Function to calculate the MMD-based loss for the Gaussian model"""

        var_eps = jnp.exp(theta[-1])   # reparametrisation to ensure variance of residual error is positive
        N = len(xsample) # be careful with the sizes!
        #y = jax.random.multivariate_normal(rng, theta[0] + theta[1]*xsample + theta[2]*xsample**2 + theta[3]*xsample**3, var_eps*jnp.eye(N)) # draw samples from the model
        y = jax.random.multivariate_normal(rng, (jnp.exp(theta[0] + theta[1]*xsample))/(1 + jnp.exp(theta[0] + theta[1]*xsample)), var_eps*jnp.eye(N))
        kyy = k_jax(xsample,y,xsample,y, self.lx, self.ly)
        kxy = k_jax(xsample,y,D[:,0],D[:,1],self.lx, self.ly)
        diag_elements = jnp.diag_indices_from(kyy)
        kyy = kyy.at[diag_elements].set(jnp.repeat(0,N))
        sum1 = jnp.sum(kyy)
        sum2 = jnp.sum(kxy)
        K = (1/(N*(N-1)))*sum1-(2/(N*N))*sum2

        return K

    def draw_samples(self):
        """Draws B samples from the nonparametric posterior"""

        # Sample Dirichlet weights for each bootstrap iteration
        weights = np.zeros((self.B, self.num_groups, self.T+self.group_sizes[0]))
        for i in range(self.num_groups):
          self.seed += 1
          group_size = self.group_sizes[i]
          dir_params = np.concatenate([(self.c/self.T)*np.ones(self.T), np.ones(group_size)])
          weights_i = dirichlet.rvs(dir_params, size=(self.B,), random_state=self.seed)
          weights[:,i,:] = weights_i
        
        #weights = np.repeat(weights[:, np.newaxis, :], self.num_groups, axis=1) # BxnxT

        # Sample pseudo-data from prior centering measure
        x_tilde = multivariate_normal.rvs(self.w_groups, (self.prior**2)*np.eye(self.num_groups), size=(self.B,self.T), random_state=self.seed) # shape BxTxn self.prior
        # x_tilde = multivariate_t.rvs(df=2, loc=self.w_groups, size=(self.B,self.T), random_state=self.seed) # shape BxTxn
        if np.isnan(x_tilde).any():
            print("NaN values encountered in x_tilde!")

        key = jax.random.PRNGKey(113)
        # Split random key into B keys
        key, *subkeys = jax.random.split(key, num=self.B+1) #B,
        # vmap over B bootstrap iterations so each term in the vectorisation is of size (n, T) for weights and (T, n) for x_tilde
        samples = vmap(self.draw_single_sample, in_axes=(0,0,0))(weights, x_tilde, jnp.array(subkeys))
        self.sample = np.array(samples)

    def minimise_MMD(self, data, weights, x_tilde, key, Nstep=700, eta=0.01):  #0.1
      """Function to minimise the MMD using adam optimisation from jax"""
      # eta: learning rate
      # Nstep: number of gradient steps

      key, subkey, key1, key2 = jax.random.split(key, num=3 + 1)
      config.update("jax_enable_x64", False)

      # objective function to feed the optimizer
      def obj_fun(theta, Ds, xs, key):
        return self.loss(key,theta,Ds,xs)

      opt_init, opt_update, get_params = optimizers.adam(step_size=eta)

      itercount = itertools.count()

      # Gradient function
      grad_fn = jit(value_and_grad(obj_fun, argnums=0)) # gradient with respect to first argument

      # Function to evaluate gradient and loss value at each step
      def step(step, opt_state, Ds, xs, key):
        value, grad = grad_fn(get_params(opt_state), Ds, xs, key)
        opt_state = opt_update(step, grad, opt_state)
        return value, opt_state

      def take_sample(w, x, y, key):
        """For one data point (x_i, y_i) draw samples from the corresponding DP"""
        
        repeated = jnp.repeat(self.w_groups, self.group_sizes)
        x = jnp.concatenate([x,repeated.reshape((self.group_sizes[0],self.num_groups))]) # Size of x is (T+1) x n
        group_size = jnp.shape(w)[1] - self.T

        def sample(w,x,key):
            # Pick (x_i, y_i) from size T+1 with corresponding weights
            inds1 = jax.random.choice(key, a=self.T+group_size, shape=(group_size,self.m), p=w)  # weights are of size nx(T+1)
            key, subkey = jax.random.split(key) # split the key for second draw
            inds2 = jax.random.choice(subkey, a=self.T+group_size, shape=(group_size,self.m), p=w)
            x1 = jnp.take(a=x, indices=inds1.squeeze())
            x2 = jnp.take(a=x, indices=inds2.squeeze())
            return x1, x2

        key, *rng_inputs = jax.random.split(key, self.num_groups+1)
        xs1, xs2 = vmap(sample, in_axes=(0,1,0))(w,x,jnp.array(rng_inputs)) # vmap over the second dimension of size n
        # print(jnp.isnan(xs1).sum())
        # print(jnp.isnan(xs2).sum())
        y = y.repeat(self.m)
        # TODO figure out sampling and sizes! 
        D = jnp.zeros((self.m*self.n,2))
        D = D.at[:,0].set(xs1.flatten())
        D = D.at[:,1].set(y)
        return D, xs2.flatten()

      # Initialization of theta for optimisation: here you can start from a fixed point or uniformly sample from a range of values
      #params = jnp.array([0.,0.,-2.]) # last parameter is variance of error on y and we have reparametrised it
      param_range = (jnp.array([-2., -2., -10.]), jnp.array([5., 5., -2.]))
      lower, upper = param_range
      params = jax.random.uniform(subkey, minval=lower, maxval=upper, shape=(self.p,))
      opt_state = opt_init(params) # initialise theta at params

      smallest_loss = 1000000
      best_theta = get_params(opt_state)

      DataSet, xsample = take_sample(weights, x_tilde, self.data[:,1], subkey)
      #TODO: try resampling DataSet and xsample within the optimisiation loop - split subkey!

      for i in range(Nstep):
        key1, subkey = jax.random.split(key1)
        # update gradient
        key2, subkey = jax.random.split(key2)
        value, opt_state = step(next(itercount), opt_state, DataSet, xsample, subkey)
        # Update smallest loss and best theta value if loss has decreased - fast version for JAX
        pred =  value < smallest_loss # Prediction is that current value of loss is smaller than smallest_loss
        def true_func(args): # if pred is true update smallest_loss and best_theta
            value, smallest_loss, best_theta, opt_state = args[0], args[1], args[2], args[3]
            smallest_loss = value
            best_theta = get_params(opt_state)
            return smallest_loss, best_theta
        def false_func(args): # if pred is false don't update smallest_loss and best_theta
            value, smallest_loss, best_theta, opt_state = args[0], args[1], args[2], args[3]
            smallest_loss = jnp.array(smallest_loss, dtype='float64')
            return smallest_loss, best_theta
        # Updates smallest loss and best theta if prediction (pred) is correct / if pred is true call true_fanc o/w call false_func
        smallest_loss, best_theta = jax.lax.cond(pred, true_func, false_func, [value, smallest_loss, best_theta, opt_state])

      return  jnp.array([best_theta[0:-1]])