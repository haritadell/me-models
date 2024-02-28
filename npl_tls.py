from utils import tls, run_ols, run_odr
import numpy as np
from scipy.stats import dirichlet, multivariate_normal, multivariate_t
import jax.numpy as jnp
from multiprocessing import Pool
import os
from joblib import Parallel, delayed
from tqdm import tqdm



# NPL class
class npl_tls():
    """This class contains functions to perform inference with the Robust-MEM method with the MMD. 
    The user supplies parameters:
        data: Data set 
        B: number of bootstrap iterations
        c: DP concentration parameter
        T: truncation limit for DP sample approximation
        seed: random seed
    """
    
    def __init__(self, data, B, c, T, seed, prior):
        self.B = B # numbr of bootstrap iterations 
        self.T = int(T) # truncation limit for DP sample approximation
        self.c = c # DP concentration parameter
        self.data = data # observed data
        self.n = len(self.data) # number of data points
        self.seed = seed # random seed            
        self.prior = prior
        self.tls_value_from_data = tls(self.data[:,0],self.data[:,1]) # TLS estimator using the observed data
        self.generator = np.random.default_rng(seed=self.seed)

    def draw_single_sample(self,i): 
        """ Draws a single sample from the NPL posterior"""

        dir_params = np.concatenate([(self.c/self.T)*np.ones(self.T), np.array([1])]) # parameters for sampling weights from the Dirichlet distribution
        weights = dirichlet.rvs(dir_params, size=(self.n,), random_state=self.generator) # (1,101) ? sample weights
        weights_resized = jnp.sqrt((1/self.n)*jnp.transpose(weights).flatten()) # sqrt of weights #.repeat(self.n)
        #x_tilde = multivariate_t.rvs(loc=self.data[:,0], df=3, size=(self.T), random_state=self.seed+i).reshape((self.T,self.n)) # sample x_tilde from the Student-t
        
        post_var = 1/((1/self.prior[0]**2) + (1/self.prior[1]**2))   
        x_tilde = multivariate_normal.rvs(post_var*(self.data[:,0]/(self.prior[0]**2)), post_var*np.eye(self.n), size=(self.T), random_state=self.generator).reshape((self.T,self.n))
        
        new_data = jnp.zeros((self.T*self.n, 2)) # Initialise matrix for weighted synthetic dataset
        new_data = new_data.at[:,0].set(x_tilde.flatten()*weights_resized[:(self.n*self.T)]) 
        new_data = new_data.at[:,1].set(self.data[:,1].repeat(self.T)*weights_resized[:(self.n*self.T)])
        
        weighted_data = jnp.zeros((self.n,2)) # Initialise matrix for weighted observed dataset 
        weighted_data = weighted_data.at[:,0].set(self.data[:,0]*weights_resized[(self.n*self.T):])
        weighted_data = weighted_data.at[:,1].set(self.data[:,1]*weights_resized[(self.n*self.T):])
        #print(weights_resized[(self.n*self.T):])
        weighted_DataSet = jnp.vstack([new_data, weighted_data]) # combine weighted synthetic and weighted observed 

        # do the same for unweighted dataset
        new_data = jnp.zeros((self.T*self.n, 2))
        new_data = new_data.at[:,0].set(x_tilde.flatten()) 
        new_data = new_data.at[:,1].set(self.data[:,1].repeat(self.T))
        unweighted_DataSet = jnp.vstack([new_data, self.data])

        # Find tls and ols estimators
        results_odr = self.minimise_odr(weighted_DataSet, unweighted_DataSet)
        results_ols = self.minimise_ols(weighted_DataSet, unweighted_DataSet)
    
        return results_odr, results_ols 

    def info(self, title):
        print(title)
        print('module name:', __name__)
        print('parent process:', os.getppid())
        print('process id:', os.getpid())
          
    def draw_samples(self):
        """Draws B samples in parallel from the nonparametric posterior"""

        results_odr = np.zeros((self.B,2)) 
        results_ols = np.zeros((self.B,2))

        temp = Parallel(n_jobs=-1, backend= 'loky')(delayed(self.draw_single_sample)(i) for i in range(self.B))
        for i in range(self.B):
            results_odr[i,:] = temp[i][0]
            results_ols[i,:] = temp[i][1]

        self.sample_odr = np.array(results_odr)
        self.sample_ols = np.array(results_ols) 
        
    def minimise_odr(self, weighted_DataSet, unweighted_DataSet):                                                                
        samples = run_odr(weighted_DataSet) 
        a_odr = np.array(samples.beta)
        b_tls = np.mean(unweighted_DataSet[:,1]) - a_odr[0]*np.mean(unweighted_DataSet[:,0])
        return np.array([a_odr[0],b_tls]) 

    def minimise_ols(self, weighted_DataSet, unweighted_DataSet):
      samples = run_ols(weighted_DataSet)
      a_ols = samples[0]
      b_ols = np.mean(unweighted_DataSet[:,1]) - a_ols*np.mean(unweighted_DataSet[:,0])
      return np.array([a_ols,b_ols])

    def minimise_TLS(self, data, weights, x_tilde, Nstep=110, eta=0.1):
        new_data = jnp.zeros((self.T*self.n, 2))
        weights_resized = jnp.sqrt(jnp.transpose(weights).flatten().repeat(self.n))
        new_data = new_data.at[:,0].set(x_tilde.flatten()*weights_resized[:(self.n*self.T)])
        new_data = new_data.at[:,1].set(self.data[:,1].repeat(self.T)*weights_resized[:(self.n*self.T)])
        weighted_data = jnp.zeros((self.n,2))
        weighted_data = weighted_data.at[:,0].set(self.data[:,0]*weights_resized[(self.n*self.T):])
        weighted_data = weighted_data.at[:,1].set(self.data[:,1]*weights_resized[(self.n*self.T):])
        DataSet = jnp.vstack([new_data, weighted_data])
        tls_value = tls(DataSet[:,0], DataSet[:,1])
        return tls_value
