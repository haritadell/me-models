from utils import sample_observed_data_berkson, mse
import time
from npl_me import npl
import numpy as np
from itertools import product
from scipy.optimize import curve_fit
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int)
args = parser.parse_args()

args.seed
folder_path = './results/'

def train_npl(params, reg_func, seed):
    n,loc_x,scale_x,scale_nu,scale_eps,B,m,c,T = params
    theta_star = np.array([1,2])
    data, x = sample_observed_data_berkson(reg_func, int(n), loc_x, scale_x, scale_nu, scale_eps, theta_star, seed)
    npl_ = npl(data,int(B),int(m), c, T, seed, lx=10, ly=10, prior=scale_nu)
    t0 = time.time()
    npl_.draw_samples()
    t1 = time.time()
    total = t1-t0
    sample = npl_.sample
    return sample, data, x

def reg_func(theta,x):
    return (np.exp(theta[0] + theta[1]*x))/(1 + np.exp(theta[0] + theta[1]*x))

# Define a nonlinear model function (e.g., a Gaussian function)
def nonlinear_model(x, a, b):
    return (np.exp(a + b*x))/(1 + np.exp(a + b*x))

n = np.array([100])
loc_x = np.array([0])
scale_x = np.array([1])
scale_nu = np.array([0.000001, 1, 2]) # Specify different values of stadnard deviation of ME (try as many as you want)
scale_eps = np.array([0.2])  # True value of \sigma^2_{\epsilon}
B = np.array([50])
m = np.array([1])
c = np.array([1]) # You can also try different values of c parameter in DP
T = np.array([100]) # Truncated limit of DP sum
theta_star = np.array([1,2])  # True parameter values
R = 1

if __name__=='__main__':
    var_nu = [i**2 for i in scale_nu]
    num_scales = len(scale_nu)
    num_cs = len(c)
    num_config = len(scale_nu) + len(c)
    posterior_samples = np.zeros((int(B[0]), len(theta_star), num_config, R))
    data_sets = np.zeros((int(n[0]), 2, num_config, R))
    x_sets = np.zeros((int(n[0]), num_config, R))
    coefficients = np.zeros((len(theta_star), num_config, R))
    
    for r in range(R):
        args.seed += 1
        for p,params in enumerate(list(product(n,loc_x,scale_x,scale_nu,scale_eps,B,m,c,T))):
            sample, data, x = train_npl(np.array(params), reg_func, args.seed)
            posterior_samples[:, :, p, r] = sample.reshape((int(B[0]), len(theta_star)))
            data_sets[:, :, p, r] = data
            x_sets[:, p, r] = x

            # Fit the nonlinear model to the data using curve_fit
            initial_guess = [0, 0]  # Initial parameter guess
            coefs, covariance = curve_fit(nonlinear_model, data[:,0], data[:,1], p0=initial_guess)
            coefficients[:, p, r] = coefs

            mses = np.zeros((2,len(theta_star)))
            stds = np.zeros((2,len(theta_star)))

            mses[0,:], stds[0,:] = mse(posterior_samples[:, :, p, r],theta_star)
            mses[1,:] = np.asarray((coefs-theta_star))**2

            print(mses, stds, p, r)
            
            
    