from utils import sample_observed_data_berkson, sample_observed_data_classical, mse, cars_dataset
import time
from npl_poly import npl
import numpy as np
from itertools import product
from scipy.optimize import curve_fit
import pandas as pd
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('seed1', type=int)
parser.add_argument('seed2', type=int)
args = parser.parse_args()

args.seed1
args.seed2
folder_path = '/dcs/pg23/u1604520/mem/results/cars/'

def reg_func(theta,x):
    return theta[0] + theta[1]*x + theta[2]*x**2

def train_npl(params, reg_func, seed1, seed2):
    scale_nu,B,m,c,T,p_ = params
    data, x = cars_dataset('cars.csv', scale_nu, seed1)
    npl_ = npl(data,int(B),int(m), c, T, int(p_), seed2, lx=1, ly=1, prior=np.array([scale_nu]), me_type='berkson')  #np.array([scale_nu, 1.0])
    t0 = time.time()
    npl_.draw_samples()
    t1 = time.time()
    total = t1-t0
    sample = npl_.sample
    return sample, data, x

# Define a nonlinear model function 
def nonlinear_model(x, a, b, c):
    return a + b*x + c*x**2

scale_nu = np.array([0.000001, 0.5, 1, 2]) # Specify different values of stadnard deviation of ME (try as many as you want)
B = np.array([500])
m = np.array([1])
c = np.array([1]) # You can also try different values of c parameter in DP
T = np.array([100]) # Truncated limit of DP sum
p_ = np.array([4]) 
R = 1

if __name__=='__main__':
    var_nu = [i**2 for i in scale_nu]
    num_scales = len(scale_nu)
    num_cs = len(c)
    num_config = len(scale_nu)*len(c)
    posterior_samples = np.zeros((int(B[0]), int(p_[0] - 1), num_config, R))
    coefficients = np.zeros((int(p_[0] - 1), num_config, R))
    
    for r in tqdm(range(R)):
        args.seed1 += 1
        args.seed2 += 1
        for p,params in enumerate(list(product(scale_nu,B,m,c,T,p_))):
            sample, data, x = train_npl(np.array(params), reg_func, args.seed1, args.seed2)
            posterior_samples[:, :, p, r] = sample.reshape((int(B[0]), int(p_[0] - 1)))
            # # Fit the nonlinear model to the data using curve_fit
            # initial_guess = [0, 4]  # Initial parameter guess
            # coefs, covariance = curve_fit(nonlinear_model, data[:,0], data[:,1], p0=initial_guess)
            # coefficients[:, p, r] = coefs
            # mses = np.zeros((2,len(theta_star)))
            # stds = np.zeros((2,len(theta_star)))
            # mses[0,:], stds[0,:] = mse(posterior_samples[:, :, p, r],theta_star)
            # mses[1,:] = np.asarray((coefs-theta_star))**2
            #print(mses)
            # np.savetxt(folder_path+f'mses_scale_nu{params[3]}_c{params[7]}_n{params[0]}_B{params[5]}_seed{args.seed}.txt', mses)
            np.savetxt(folder_path+f'sample_scale_nu{params[0]}_c{params[3]}_B{params[1]}_seed1{args.seed1}_seed2{args.seed2}.txt', posterior_samples[:, :, p, r])
            #np.savetxt(folder_path+f'data_scale_nu{params[3]}_c{params[7]}_n{params[0]}_B{params[5]}_seed{args.seed}.txt', data)
            #np.savetxt(folder_path+f'x_scale_nu{params[3]}_c{params[7]}_n{params[0]}_B{params[5]}_seed{args.seed}.txt', x)
            
            
    