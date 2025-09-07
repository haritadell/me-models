from utils import tls, run_ols, run_odr
import numpy as np
from scipy.stats import dirichlet, multivariate_normal, multivariate_t
import jax.numpy as jnp
from multiprocessing import Pool
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import jax.numpy.linalg as la



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
        # self.data = np.zeros((self.n,2))
        # self.data[:,0] = data[:,0] - np.mean(data[:,0])
        # self.data[:,1] = data[:,1] - np.mean(data[:,1])
        self.tls_value_from_data = run_odr(self.data).beta #tls(self.data[:,0],self.data[:,1]) # TLS estimator using the observed data
        #print(self.tls_value_from_data, 'tls1')
        #print(tls(self.data[:,0],self.data[:,1]), 'tls2')
        #print(self.tls_value_from_data)
        self.generator = np.random.default_rng(seed=self.seed)

    def draw_single_sample(self,i): 
        """ Draws a single sample from the NPL posterior"""
        generator = np.random.default_rng(self.seed*100+i)
        dir_params = np.concatenate([(self.c/self.T)*np.ones(self.T), np.array([1])]) # parameters for sampling weights from the Dirichlet distribution
        weights = dirichlet.rvs(dir_params, size=(self.n,), random_state=generator) # (1,101)  sample weights
        #weights_resized = (1/self.n)*jnp.sqrt(jnp.transpose(weights).flatten()) # sqrt of weights #.repeat(self.n) 
        weights_resized = jnp.sqrt(jnp.transpose(weights).flatten())
        #x_tilde = multivariate_t.rvs(loc=self.data[:,0], df=3, size=(self.T), random_state=self.seed+i).reshape((self.T,self.n)) # sample x_tilde from the Student-t
         
        post_var = 1/((1/self.prior[0]**2) + (1/self.prior[1]**2))   
        post_mean = 10*(self.prior[0]**2)/(self.prior[1]**2 + self.prior[0]**2) + (self.data[:,0]*(self.prior[1]**2)/(self.prior[1]**2 + self.prior[0]**2))
        x_tilde = multivariate_normal.rvs(post_mean, post_var*np.eye(self.n), size=(self.T), random_state=generator).reshape((self.T,self.n))
        
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
        results_odr = self.minimise_odr(weighted_DataSet, unweighted_DataSet, weights, x_tilde)
        #results_ols = self.minimise_ols(weights, x_tilde)
    
        return results_odr #results_ols,

    def info(self, title):
        print(title)
        print('module name:', __name__)
        print('parent process:', os.getppid())
        print('process id:', os.getpid())
          
    def draw_samples(self):
        """Draws B samples in parallel from the nonparametric posterior"""

        results_odr = np.zeros((self.B,2)) 
        results_ols = np.zeros((self.B,2))

        temp = Parallel(n_jobs=-1, backend="threading")(delayed(self.draw_single_sample)(i) for i in range(self.B))
        for i in range(self.B):
            results_odr[i,:] = temp[i] #[1]
            #results_ols[i,:] = temp[i][0]
        #print(results_ols)
        self.sample_odr = np.array(results_odr)
        #self.sample_ols = np.array(results_ols) 
        
    def minimise_odr(self, weighted_DataSet, unweighted_DataSet, weights, x_tilde):                                                                
        # samples = run_odr(weighted_DataSet) 
        # slope_odr = np.array(samples.beta)
        # #print(slope_odr)
        # #inter_odr = np.mean(unweighted_DataSet[:,1]) - slope_odr[1]*np.mean(unweighted_DataSet[:,0])
        weights_resized = jnp.transpose(weights).flatten() 
        
        x_bar = (1/self.n)*jnp.sum(weighted_DataSet[:,0])
        y_bar = (1/self.n)*jnp.sum(weighted_DataSet[:,1])
        
        combined_data = jnp.zeros(((self.T+1)*self.n, 2)) # Initialise matrix for weighted synthetic dataset
        combined_data = combined_data.at[:(self.n*self.T), 0].set((x_tilde.flatten() - x_bar)*weights_resized[:(self.n*self.T)])
        combined_data = combined_data.at[(self.n*self.T):, 0].set((self.data[:,0] - x_bar)*weights_resized[(self.n*self.T):]) 
        combined_data = combined_data.at[:(self.n*self.T), 1].set((self.data[:,1].repeat(self.T) - y_bar)*weights_resized[:(self.n*self.T)])
        combined_data = combined_data.at[(self.n*self.T):, 1].set((self.data[:,1] - y_bar)*weights_resized[(self.n*self.T):]) 
        
        
        # inter_odr = y_bar - slope_odr[1]*x_bar
        # #print(slope_odr)
        # print(np.array([inter_odr, slope_odr[1]]), 'diko')
        
        X_ = combined_data[:,0]
        y_ = combined_data[:,1] 
        d = 1
    
        Z = jnp.vstack((X_.T,y_)).T #Z is (n,2)
        #print(Z)
        U, s, Vt = la.svd(Z, full_matrices=True)
    
        V = Vt.T
        Vxy = V[:d,d:]
        Vyy = V[d:,d:]
        a_tls = - Vxy  / Vyy # total least squares soln
        b_tls = y_bar - a_tls[0][0]*x_bar
        #print(np.array([b_tls, a_tls[0][0]]), 'diko')
        return np.array([b_tls, a_tls[0][0]]) #np.array([inter_odr, slope_odr[1]]) 

    def minimise_ols(self, weights, x_tilde):
    #   samples = run_ols(weighted_DataSet)
    #   a_ols = samples[0]
    #   #b_ols = np.mean(unweighted_DataSet[:,1]) - a_ols*np.mean(unweighted_DataSet[:,0])
    #   b_ols = np.mean(self.data[:,1]) - a_ols*np.mean(self.data[:,0])
        lr = LinearRegression(fit_intercept=True)
        lr.fit(self.data[:,0].reshape(-1, 1), self.data[:,1].reshape(-1, 1))
        ww = lr.coef_
        bb = lr.intercept_
        #print(bb, ww[0], 'lr')
        weights_resized = jnp.transpose(weights).flatten()  #(1/self.n)*
        new_data = jnp.zeros((self.T*self.n, 2)) # Initialise matrix for weighted synthetic dataset
        new_data = new_data.at[:,0].set(x_tilde.flatten()*weights_resized[:(self.n*self.T)]) 
        new_data = new_data.at[:,1].set(self.data[:,1].repeat(self.T)*weights_resized[:(self.n*self.T)])
        weighted_data = jnp.zeros((self.n,2)) # Initialise matrix for weighted observed dataset 
        weighted_data = weighted_data.at[:,0].set(self.data[:,0]*weights_resized[(self.n*self.T):])
        weighted_data = weighted_data.at[:,1].set(self.data[:,1]*weights_resized[(self.n*self.T):])
        #print(weights_resized[(self.n*self.T):])
        weighted_DataSet = jnp.vstack([new_data, weighted_data]) # combine weighted synthetic and weighted observed 
        z_bar = (1/self.n)*jnp.sum(weighted_DataSet[:,0])
        y_bar = (1/self.n)*jnp.sum(weighted_DataSet[:,1])
        
        # yz = jnp.zeros((self.T +1)*self.n)
        # yz = yz.at[:(self.n*self.T)].set(x_tilde.flatten()*(self.data[:,1].repeat(self.T)*weights_resized[:(self.n*self.T)]))
        # yz = yz.at[(self.n*self.T):].set(self.data[:,0]*self.data[:,1]*weights_resized[(self.n*self.T):])
        # zz = jnp.zeros((self.T +1)*self.n)
        # zz = zz.at[:(self.n*self.T)].set(x_tilde.flatten()**2*weights_resized[:(self.n*self.T)])
        # zz = zz.at[(self.n*self.T):].set(self.data[:,0]**2*weights_resized[(self.n*self.T):])
        # z_bar = (1/self.n)*jnp.sum(weighted_DataSet[:,0])
        # y_bar = (1/self.n)*jnp.sum(weighted_DataSet[:,1])
        # slope = (jnp.sum(zz) - self.n*y_bar*z_bar)/(jnp.sum(zz) - self.n*z_bar**2)
        # intercept = y_bar - slope*z_bar
        
        combined_data = jnp.zeros(((self.T+1)*self.n, 2)) # Initialise matrix for weighted synthetic dataset
        combined_data = combined_data.at[:(self.n*self.T), 0].set(x_tilde.flatten())
        combined_data = combined_data.at[(self.n*self.T):, 0].set(self.data[:,0]) 
        combined_data = combined_data.at[:(self.n*self.T), 1].set(self.data[:,1].repeat(self.T))
        combined_data = combined_data.at[(self.n*self.T):, 1].set(self.data[:,1]) 
        
        num = jnp.sum(weights_resized*(combined_data[:,0] - z_bar)*(combined_data[:,1] - y_bar))
        den = jnp.sum(weights_resized*(combined_data[:,0] - z_bar)**2)
        
        slope = num/den
        intercept = y_bar - slope*z_bar
        
        ##### OLS
        x_mean = np.mean(self.data[:,0])
        y_mean = np.mean(self.data[:,1])
    
        # Calculate slope (beta1)
        numerator = np.sum((self.data[:,0] - x_mean) * (self.data[:,1] - y_mean))
        # print(jnp.sum((weighted_DataSet[:,0] - z_bar)*(weighted_DataSet[:,1] - y_bar)))
        # print(numerator)
        denominator = np.sum((self.data[:,0] - x_mean) ** 2)
        # print(jnp.sum((weighted_DataSet[:,0] - z_bar)**2))
        # print(denominator, 'hii')
        beta1 = numerator / denominator
        beta0 = y_mean - beta1 * x_mean
        #print(beta0, beta1, 'ols')
    
        
        return np.array([intercept, slope])

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
