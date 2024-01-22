# NPL.py
import os
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
from utils import k_jax, make_design_matrix
import itertools
import numpy as np
from scipy.stats import dirichlet, multivariate_normal, multivariate_t
import scipy.spatial.distance as distance
import jax.numpy as jnp
import jax
from jax import vmap, value_and_grad, jit, pmap
from jax.config import config
#from jax.experimental import optimizers
from sklearn.linear_model import LinearRegression
import scipy.odr as odr
from jax.example_libraries import optimizers
import math

#jax.local_device_count() 

# NPL class
class npl():
    """This class contains functions to perform NPL inference for any of the models in models.py. 
    The user supplies parameters:
        X: Data set 
        B: number of bootstrap iterations
        m: number of points sampled from P_\theta at each optim. iteration
        p: number of unknown parameters
        l: lengthscale of gaussian kernel
        model: model in the form as in models.py
        model_name: name set to 'gaussian' or 'gandk' or 'toggle_switch'
    """
    
    def __init__(self, data, B, m, c, T, seed, knots, lx, ly, prior):
        #self.model = model
        self.B = int(B)  
        self.m = m
        self.T = int(T) # truncation limit for DP sample approximation
        self.c = c# DP concentration parameter
        self.data = data  
        #self.n, self.d = self.data[:,0].shape {}
        self.n = len(self.data[:,0])
        #self.d = 1  
        self.seed = seed
        self.prior = prior
        self.lx = lx
        self.ly = ly
        if self.lx == -1:
            self.lx = np.sqrt((1/2)*np.median(distance.cdist(self.data[:,0].reshape((self.n,1)),self.data[:,0].reshape((self.n,1)),'sqeuclidean')))
        if self.ly == -1:
            self.ly = np.sqrt((1/2)*np.median(distance.cdist(self.data[:,1].reshape((self.n,1)),self.data[:,1].reshape((self.n,1)),'sqeuclidean')))
        self.knots = knots
        self.K = len(self.knots)
        def fn_to_scan(B, j):
          cond = (self.data[:,0]>=self.knots[j])&(self.data[:,0]<=self.knots[j+1])
          act_x = jnp.where(cond,self.data[:,0],0)
          resc_x = (act_x-self.knots[j])/(self.knots[j+1]-self.knots[j])
          B = B.at[:,j].set(jnp.where(cond,(1/2)*(1-resc_x)**2,B[:,j]))
          B = B.at[:,j+1].set(jnp.where(cond,-(resc_x**2)+resc_x+1/2,B[:,j+1]))
          B = B.at[:,j+2].set(jnp.where(cond,(resc_x**2)/2,B[:,j+2]))

          return B, j+1
        self.Z_data = jax.lax.scan(fn_to_scan, jnp.zeros((self.n,self.K+1)), jnp.arange(self.K-1))[0]  

    def draw_single_sample(self, weights, x_tilde, key, ind): #, x_stars): #
        """ Draws a single sample from the nonparametric posterior specified via
        the model and the data X"""
        theta_j = self.minimise_MMD(self.data, weights, x_tilde, key, ind)
             
        return theta_j
    
    def loss(self, rng, theta, D, xsample):
        var_eps = jnp.exp(theta[-1])   # variance of error on y
        u = theta[:-1]
        N = len(xsample)
        K = 10
        def fn_to_scan(B, j):
          cond = (xsample>=self.knots[j])&(xsample<=self.knots[j+1])
          act_x = jnp.where(cond,xsample,0)
          resc_x = (act_x-self.knots[j])/(self.knots[j+1]-self.knots[j])
          B = B.at[:,j].set(jnp.where(cond,(1/2)*(1-resc_x)**2,B[:,j]))
          B = B.at[:,j+1].set(jnp.where(cond,-(resc_x**2)+resc_x+1/2,B[:,j+1]))
          B = B.at[:,j+2].set(jnp.where(cond,(resc_x**2)/2,B[:,j+2]))

          return B, j+1

        Z = jax.lax.scan(fn_to_scan, jnp.zeros((N,K+1)), jnp.arange(K-1))[0]
        X = make_design_matrix(xsample) # X is Nx3, Zs are NxK (22)
        y = jax.random.multivariate_normal(rng, jnp.matmul(Z,u) , var_eps*jnp.eye(N)) 
        
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
        dir_params = np.concatenate([(self.c/self.T)*np.ones(self.T), np.array([1])])
        weights = dirichlet.rvs(dir_params, size=(self.B,self.n), random_state=self.seed+10)
        post_var = 1/((1/self.prior**2) + (1/2))   
        x_tilde = multivariate_normal.rvs(post_var*(self.data[:,0]/(2)), post_var*np.eye(self.n), size=(self.B,self.T), random_state=self.seed)
        key = jax.random.PRNGKey(113)
        key, *subkeys = jax.random.split(key, num=self.B+1)
        samples = vmap(self.draw_single_sample, in_axes=(0,0,0,0))(weights, x_tilde, jnp.array(subkeys), jnp.arange(self.B)) # vmap over B so each term in the parallelisation is of size (n, T) for weights and (T, n) for x_tilde
        self.sample = np.array(samples) 
    

    def minimise_MMD(self, data, weights, x_tilde, key, ind, Nstep=100, eta=0.5): 
      """Function to minimise the MMD using adam optimisation from jax"""
      key, subkey = jax.random.split(key)
      config.update("jax_enable_x64", False)
      def obj_fun(theta, Ds, xs, key):
        return self.loss(key,theta,Ds,xs)
            
      opt_init, opt_update, get_params = optimizers.adam(step_size=eta) # Initialise ADAM optimiser
      
      itercount = itertools.count()

      grad_fn = jit(value_and_grad(obj_fun, argnums=0)) # gradient function with respect to first argument of obj_fun (theta) 
        
      def step(step, opt_state, Ds, xs, key):
        value, grad = grad_fn(get_params(opt_state), Ds, xs, key)
        opt_state = opt_update(step, grad, opt_state) 
        return value, opt_state
        
      key1 = jax.random.PRNGKey(11)
      key2 = jax.random.PRNGKey(13)

      def take_sample(w, x, y, key):
        """Add text here"""
        x = jnp.concatenate([x,self.data[:,0].reshape((1,self.n))]) # T+1 x n

        def sample(w,x,key):
            inds1 = jax.random.choice(key, a=self.T+1, shape=(self.m,), p=w) # Pick m from size T+1 with relevant weights / weights are nx(T+1)
            key, subkey = jax.random.split(key)
            inds2 = jax.random.choice(subkey, a=self.T+1, shape=(self.m,), p=w)
            x1 = jnp.take(a=x, indices=inds1)
            x2 = jnp.take(a=x, indices=inds2) 
            return x1, x2
          
        key, *rng_inputs = jax.random.split(key, self.n+1)
        # here we vmap over n - xs_ are mxn - ws,xs / so here for each n I am sampling one from T+1 values and I do this twice
        # (one for each argument of the MMD objective function)
        xs1, xs2 = vmap(sample, in_axes=(0,1,0))(w,x,jnp.array(rng_inputs)) 

        y = y.repeat(self.m)
        D = jnp.zeros((self.m*self.n,2))
        D = D.at[:,0].set(xs1.flatten())
        D = D.at[:,1].set(y)

        return D, xs2.flatten() 
        
      # Explain the initialisation I do here in terms of the pseudo inverse for Y = X / explain here and in paper! Make sure it makes sense
      inv = jnp.linalg.pinv(self.Z_data)
      u_init = jnp.matmul(inv,self.data[:,0])
      params = jnp.concatenate([u_init,jnp.array([-10.])]) 
      print(params)

      smallest_loss = 1000000
      K = 10 # parameter as in Sarkar et al.

      n_optimized_locations = 1 
      list_of_thetas = jnp.zeros((n_optimized_locations,12)) # Initialise list of thetas depending on the number of locations   
      for j in range(n_optimized_locations):
        opt_state = opt_init(params)
        smallest_loss = 1000
        best_theta = get_params(opt_state)
        for i in range(Nstep):    
          key1, subkey = jax.random.split(key1)
          DataSet, xsample = take_sample(weights, x_tilde, self.data[:,1], subkey)
          
          # update 
          key2, subkey = jax.random.split(key2)
          value, opt_state = step(next(itercount), opt_state, DataSet, xsample, subkey) #

          # Check if the loss is getting smaller, if it is then update best theta 
          pred =  value < smallest_loss
          def true_func(args):
              value, smallest_loss, best_theta, opt_state = args[0], args[1], args[2], args[3]
              smallest_loss = value
              best_theta = get_params(opt_state)
              return smallest_loss, best_theta
          def false_func(args):
              value, smallest_loss, best_theta, opt_state = args[0], args[1], args[2], args[3]
              smallest_loss = jnp.array(smallest_loss, dtype='float64')
              return smallest_loss, best_theta
          smallest_loss, best_theta = jax.lax.cond(pred, true_func, false_func, [value, smallest_loss, best_theta, opt_state])
          print(best_theta)  
              
    #     list_of_thetas = list_of_thetas.at[j,:].set(best_theta)

    #   DataSet, x_to_use = take_sample(weights,x_tilde,self.data[:,1],key1)  
    #   losses = []
    #   seed = 12
    #   rng = jax.random.PRNGKey(seed)
    #   rng, *rng_inputs = jax.random.split(rng, num=n_optimized_locations + 1)
  
    #   for l,t in enumerate(list_of_thetas):
    #      losses.append(self.loss(jnp.array(rng_inputs)[l,:],t,DataSet, DataSet[:,0]))
    #   ind_ = jnp.argmin(jnp.asarray(losses)) # get the index of smallest loss 
    #   best_theta = list_of_thetas[ind]    
    #   print(ind_)  
      a_mmd = best_theta[0:-1] #get_params(opt_state)[0] 
      
      return  jnp.array([a_mmd])