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
from scipy.optimize import curve_fit
from utils import k_comp, make_design_matrices, create_knots, k_jax

# NPL class
class npl_class():
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
        self.lx = lx
        self.ly = ly
        self.prior = prior
        self.generator = np.random.default_rng(seed=self.seed)
        if self.lx == -1:
            self.lx = np.sqrt((1/2)*np.median(distance.cdist(self.data[:,0].reshape((self.n,1)),self.data[:,0].reshape((self.n,1)),'sqeuclidean')))
        if self.ly == -1:
            self.ly = np.sqrt((1/2)*np.median(distance.cdist(self.data[:,1].reshape((self.n,1)),self.data[:,1].reshape((self.n,1)),'sqeuclidean')))
        

    def draw_single_sample(self, weights, x_tilde, key, ind):
        """ Draws a single sample from the nonparametric posterior specified via
        the model and the data X"""

        theta_j, knots_j = self.minimise_MMD(self.data, weights, x_tilde, key, ind)
             
        return theta_j, knots_j

    def loss(self, rng, theta, D, xsample, knots):
        var_eps = jnp.exp(theta[-1])   # variance of error on y
        beta = theta[:4]
        u_0 = theta[4:34]
        u_1 = theta[34:-1]
        
        N = len(xsample)

        def create_z(xi,knotsk):
          pred = xi > knotsk
          res = jax.lax.select(pred,1,0)
          return (xi-knotsk)*res
        map1 = vmap(create_z, in_axes=(0,None), out_axes=0)
        map2 = vmap(map1, in_axes=(None,0), out_axes=1)
        z = map2(xsample,knots)

        X, Z_0, Z_1 = make_design_matrices(xsample,z) # X is Nx3, Zs are NxK (22)

        y = jax.random.multivariate_normal(rng, jnp.matmul(X,beta)+jnp.matmul(Z_0,u_0)+jnp.matmul(Z_1,u_1) , var_eps*jnp.eye(N))

        kyy = k_comp(xsample,y,xsample,y)
        kxy = k_comp(xsample,y,D[:,0],D[:,1])
        # kyy = k_jax(xsample,y,xsample,y,self.lx,self.ly)
        # kxy = k_jax(xsample,y,D[:,0],D[:,1],self.lx,self.ly)
        diag_elements = jnp.diag_indices_from(kyy)
        kyy = kyy.at[diag_elements].set(jnp.repeat(0,N))
        sum1 = jnp.sum(kyy)
        sum2 = jnp.sum(kxy)
        K = (1/(N*(N-1)))*sum1-(2/(N*N))*sum2
        
        return K

    def draw_samples(self):
        """Draws B samples from the nonparametric posterior"""
        dir_params = np.concatenate([(self.c/self.T)*np.ones(self.T), np.array([1])])
        weights = dirichlet.rvs(dir_params, size=(self.B,self.n), random_state=self.generator)
        # Here self.prior is prior variance of marginal distribution of X
        post_var = 1/((1/self.prior**2) + (1/0.35))   
        post_mean = 0.35/(self.prior**2 + 0.35)*0 + (self.prior**2/(0.35 + self.prior**2))*self.data[:,0]
        x_tilde = multivariate_normal.rvs(post_mean, post_var*np.eye(self.n), size=(self.B,self.T), random_state=self.generator)
        key = jax.random.PRNGKey(self.seed)
        key, *subkeys = jax.random.split(key, num=self.B+1)
        # init_params = vmap(self.best_init_params)(jnp.arange(self.B)).reshape((self.B,65))
        # print(init_params.shape)
        samples, knots_samples = vmap(self.draw_single_sample, in_axes=(0,0,0,0))(weights, x_tilde, jnp.array(subkeys), jnp.arange(self.B)) # vmap over B so each term in the parallelisation is of size (n, T) for weights and (T, n) for x_tilde
        self.sample = np.array(samples) 
        self.knots_sample = np.array(knots_samples)

    def best_init_params(self, key):
        """Function to find optimisation starting point"""
        
        def sample_theta_init(rng):
          lower_b = jnp.concatenate([jnp.array([0, 0, 0, 0]),(-0.1)*jnp.ones(30), (-0.1)*jnp.ones(30),jnp.array([-10])]) #0-2, -0.1-0.1
          upper_b = jnp.concatenate([jnp.array([2, 2, 2, 2]),(0.1)*jnp.ones(30), (0.1)*jnp.ones(30),jnp.array([-1])])
          param_range = (lower_b, upper_b) 
          lower, upper = param_range
          params = jax.random.uniform(rng, minval=lower, maxval=upper, shape=lower.shape)
          # params = jnp.zeros(65)
          # params = params.at[:64].set(jax.random.multivariate_normal(rng, mean=jnp.ones(64), cov=jnp.eye(64)))
          # params = params.at[-1].set(jax.random.uniform(rng, minval=-10, maxval=-1))
          return params  

        n_initial_locations = 100 #1000
        n_optimized_locations = 1

        key, *rng_inputs = jax.random.split(key, num=n_initial_locations*3 + 1)
        init_thetas = vmap(sample_theta_init)(jnp.array(rng_inputs[:n_initial_locations]))
        #rng, *rng_inputs = jax.random.split(rng, num=n_initial_locations + 1)
        
        K = 30
        knots = vmap(create_knots, in_axes=(None,None,0))(self.data[:,0],K,jnp.arange(K))
        init_losses = []
        for i, t in enumerate(init_thetas):
            inds = jax.random.choice(rng_inputs[n_initial_locations+i], a=self.n, shape=(self.n,), p=(1/self.n)*jnp.ones(self.n))
            xsample = jnp.take(a=self.data[:,0], indices=inds)
            init_losses.append(self.loss(rng_inputs[2*n_initial_locations+i],t,self.data,self.data[:,0],knots)) #xsample

        best_init_params = init_thetas[jnp.argsort(jnp.asarray(init_losses))[:n_optimized_locations]]

        return best_init_params

    def minimise_MMD(self, data, weights, x_tilde, key, ind, Nstep=700, eta=0.001): #0.1  
      """Function to minimise the MMD using adam optimisation from jax"""
      # eta: learning rate
      # Nstep: number of gradient steps

      key, key1, key2, key3, key4 = jax.random.split(key, num=4 + 1)
      config.update("jax_enable_x64", True)

      # objective function to feed the optimizer
      def obj_fun(theta, Ds, xs, knots, key):
        return self.loss(key, theta, Ds, xs, knots)

      opt_init, opt_update, get_params = optimizers.adam(step_size=eta)

      itercount = itertools.count()

      # Gradient function
      grad_fn = jit(value_and_grad(obj_fun, argnums=0)) # gradient with respect to first argument

      # Function to evaluate gradient and loss value at each step
      def step(step, opt_state, Ds, xs, knots, key):
        value, grad = grad_fn(get_params(opt_state), Ds, xs, knots, key)
        opt_state = opt_update(step, grad, opt_state)
        return value, opt_state

      def take_sample(w, x, y, key):
        """For one data point (x_i, y_i) draw samples from the corresponding DP"""
        
        x = jnp.concatenate([x,self.data[:,0].reshape((1,self.n))])

        def sample(w,x,key):
            # Pick (x_i, y_i) from size T+1 with corresponding weights
            inds1 = jax.random.choice(key, a=self.T+1, shape=(self.m,), p=w) # Pick m from size T+1 with relevant weights / weights are nx(T+1)
            key, subkey = jax.random.split(key)
            inds2 = jax.random.choice(subkey, a=self.T+1, shape=(self.m,), p=w)
            x1 = jnp.take(a=x, indices=inds1.squeeze())
            x2 = jnp.take(a=x, indices=inds2.squeeze())
            return x1, x2

        key, *rng_inputs = jax.random.split(key, self.n+1)
        xs1, xs2 = vmap(sample, in_axes=(0,1,0))(w,x,jnp.array(rng_inputs)) # vmap over the second dimension of size n
        # print(jnp.isnan(xs1).sum())
        # print(jnp.isnan(xs2).sum())
        y = y.repeat(self.m)
        D = jnp.zeros((self.m*self.n,2))
        D = D.at[:,0].set(xs1.flatten())
        D = D.at[:,1].set(y)
        return D, xs2.flatten()

      # Initialization of theta for optimisation
      params = self.best_init_params(key1) #jax.random.PRNGKey(self.seed)
      # lower_b = jnp.concatenate([jnp.array([-1., -1., -1., -1.]),(-1)*jnp.ones(60),jnp.array([-10])]) 
      # upper_b = jnp.concatenate([jnp.array([1., 1., 1., 1.]),(1)*jnp.ones(60),jnp.array([-1])])
      # param_range = (lower_b, upper_b) 
      # lower, upper = param_range
      # params = jax.random.uniform(subkey, minval=lower, maxval=upper, shape=lower.shape)
      smallest_loss = 1000000
      K = 30
      knots_b = jnp.zeros(K)

      n_optimized_locations = 1
      list_of_thetas = jnp.zeros((n_optimized_locations,65)) 
      list_of_knots = jnp.zeros((n_optimized_locations,30))  

      DataSet, xsample = take_sample(weights, x_tilde, self.data[:,1], key2)
      key3, *rng_inputs3 = jax.random.split(key3, num=Nstep + 1)
      
      # Optimisation loop 
      for j in range(n_optimized_locations):
        opt_state = opt_init(params[j,:]) 
        smallest_loss = 1000
        best_theta = get_params(opt_state)
        for i in range(Nstep):    
          # compute and save knots once for this particular xsample
          knots = vmap(create_knots, in_axes=(None,None,0))(xsample,K,jnp.arange(K))
          # update  
          value, opt_state = step(next(itercount), opt_state, DataSet, xsample, knots, rng_inputs3[i]) 
          pred =  value < smallest_loss
          def true_func(args):
              value, smallest_loss, best_theta, opt_state, knots, knots_b = args[0], args[1], args[2], args[3], args[4], args[5]
              smallest_loss = value
              best_theta = get_params(opt_state)
              knots_b = knots
              return smallest_loss, best_theta, knots_b 
          def false_func(args):
              value, smallest_loss, best_theta, opt_state, knots, knots_b = args[0], args[1], args[2], args[3], args[4], args[5]
              smallest_loss = jnp.array(smallest_loss, dtype='float64')
              return smallest_loss, best_theta, knots_b
          smallest_loss, best_theta, knots_b = jax.lax.cond(pred, true_func, false_func, [value, smallest_loss, best_theta, opt_state, knots, knots_b])
              
        list_of_thetas = list_of_thetas.at[j,:].set(best_theta)
        list_of_knots = list_of_knots.at[j,:].set(knots_b)
      DataSet, x_to_use = take_sample(weights,x_tilde,self.data[:,1],key1)  
      losses = []
      rng, *rng_inputs = jax.random.split(key4, num=n_optimized_locations + 1)
  
      for l,t in enumerate(list_of_thetas):
         losses.append(self.loss(jnp.array(rng_inputs)[l,:],t,DataSet, DataSet[:,0],list_of_knots[l,:]))
      indx = jnp.argmin(jnp.asarray(losses))
      best_theta = list_of_thetas[indx]    
      knots_b = list_of_knots[indx] 
      a_mmd = best_theta[0:-1] 

      return  jnp.array([a_mmd]), knots_b