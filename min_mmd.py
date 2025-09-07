import os
import itertools
import numpy as np
from scipy.stats import dirichlet, multivariate_normal, multivariate_t
import scipy.spatial.distance as distance
import jax.numpy as jnp
import jax
from jax import vmap, value_and_grad, jit, pmap
# from jax.config import config
from sklearn.linear_model import LinearRegression
import scipy.odr as odr
from jax.example_libraries import optimizers
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import k_jax

class MMD():

    def __init__(self, data, p, lx, ly, seed):
        self.data = data
        self.n = len(self.data[:,0]) # number of observations
        self.seed = seed
        self.p = p # number of unknown parameters
        self.lx = lx
        self.ly = ly
        if self.lx == -1:
            self.lx = np.sqrt((1/2)*np.median(distance.cdist(self.data[:,0].reshape((self.n,1)),self.data[:,0].reshape((self.n,1)),'sqeuclidean')))
        if self.ly == -1:
            self.ly = np.sqrt((1/2)*np.median(distance.cdist(self.data[:,1].reshape((self.n,1)),self.data[:,1].reshape((self.n,1)),'sqeuclidean')))
        self.generator = np.random.default_rng(seed=self.seed)
        self.key = jax.random.PRNGKey(self.seed)

    def loss(self, rng, theta):
        """Function to calculate the MMD-based loss for the Gaussian model"""

        var_eps = jnp.exp(theta[-1])   # reparametrisation to ensure variance of residual error is positive
        # y = jax.random.multivariate_normal(rng, theta[0] + theta[1]*self.data[:,0] + theta[2]*self.data[:,0]**2, var_eps*jnp.eye(self.n)) # draw samples from the model
        y = jax.random.multivariate_normal(rng, (jnp.exp(theta[0] + theta[1]*self.data[:,0]))/(1 + jnp.exp(theta[0] + theta[1]*self.data[:,0])), var_eps*jnp.eye(self.n)) 
        kyy = k_jax(self.data[:,0],y,self.data[:,0],y, self.lx, self.ly)
        kxy = k_jax(self.data[:,0],y,self.data[:,0],self.data[:,1],self.lx, self.ly)
        diag_elements = jnp.diag_indices_from(kyy)
        kyy = kyy.at[diag_elements].set(jnp.repeat(0,self.n))
        sum1 = jnp.sum(kyy)
        sum2 = jnp.sum(kxy)
        K = (1/(self.n*(self.n-1)))*sum1-(2/(self.n*self.n))*sum2

        return K


    def minimise_MMD(self, Nstep=700, eta=0.1):  #0.01
      """Function to minimise the MMD using adam optimisation from jax"""
      # eta: learning rate
      # Nstep: number of gradient steps

      self.key, key1, key2, key3 = jax.random.split(self.key, num=3 + 1)
      del self.key
      # config.update("jax_enable_x64", True)

      # objective function to feed the optimizer
      def obj_fun(theta, key):
        return self.loss(key,theta)

      opt_init, opt_update, get_params = optimizers.adam(step_size=eta)

      itercount = itertools.count()

      # Gradient function
      grad_fn = jit(value_and_grad(obj_fun, argnums=0)) # gradient with respect to first argument

      # Function to evaluate gradient and loss value at each step
      def step(step, opt_state, key):
        value, grad = grad_fn(get_params(opt_state), key)
        opt_state = opt_update(step, grad, opt_state)
        return value, opt_state
    
      param_range = (jnp.array([0., 0., -10.]), jnp.array([4., 4., -2.]))
      #param_range = (jnp.array([-1., -1., -10.]), jnp.array([4., 4., -2.]))
      lower, upper = param_range
      params = jax.random.uniform(key1, minval=lower, maxval=upper, shape=(self.p,))
      del key1
      opt_state = opt_init(params) # initialise theta at params

      smallest_loss = 1000000
      best_theta = get_params(opt_state)

      key3, *rng_inputs3 = jax.random.split(key3, num=Nstep + 1)
      del key3
      
      for i in range(Nstep):
        # update gradient
        value, opt_state = step(next(itercount), opt_state, rng_inputs3[i])
        # Update smallest loss and best theta value if loss has decreased - fast version for JAX
        pred =  value < smallest_loss # Prediction is that current value of loss is smaller than smallest_loss
        def true_func(args): # if pred is true update smallest_loss and best_theta
            value, smallest_loss, best_theta, opt_state = args[0], args[1], args[2], args[3]
            smallest_loss = value
            best_theta = get_params(opt_state)
            return smallest_loss, best_theta
        def false_func(args): # if pred is false don't update smallest_loss and best_thet
            value, smallest_loss, best_theta, opt_state = args[0], args[1], args[2], args[3]
            smallest_loss = jnp.array(smallest_loss, dtype='float64')
            return smallest_loss, best_theta
        # Updates smallest loss and best theta if prediction (pred) is correct / if pred is true call true_fanc o/w call false_func
        smallest_loss, best_theta = jax.lax.cond(pred, true_func, false_func, [value, smallest_loss, best_theta, opt_state])

      return  jnp.array([best_theta[0:-1]])