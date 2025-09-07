from utils import sample_observed_data_berkson, sample_observed_data_classical, mse
import time
from npl_me import npl
import numpy as np
from itertools import product
from scipy.optimize import curve_fit
import pandas as pd
from tqdm import tqdm
from min_mmd import MMD
from timeit import default_timer as timer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('seed1', type=int)
parser.add_argument('seed2', type=int)
args = parser.parse_args()

args.seed1
args.seed2
folder_path = '/dcs/pg23/u1604520/mem/results/times/exponential_classical_unifstart/'

def reg_func(theta,x):
    return (np.exp(theta[0] + theta[1]*x))/(1 + np.exp(theta[0] + theta[1]*x))

def train_npl(params, reg_func, seed1, seed2):
    n,loc_x,scale_x,scale_nu,scale_eps,B,m,c,T,p_ = params
    theta_star = np.array([1,2]) 
    data, x = sample_observed_data_classical(reg_func, int(n), loc_x, scale_x, scale_nu, scale_eps, theta_star, seed1) 
    npl_ = npl(data,int(B),int(m), c, T, int(p_), seed2, lx=100, ly=100, prior=np.array([0.2, 1.0]), me_type='classical')  #np.array([scale_nu, 1.0])
    start = timer()
    npl_.draw_samples()
    end = timer()
    total = end - start 
    sample = npl_.sample
    #mmd = MMD(data, len(theta_star)+1, lx=-1, ly=-1, seed=seed2)
    #mmd_est = sample #mmd.minimise_MMD()
    return sample, data, x, total #mmd_est,

# Define a nonlinear model function 
def nonlinear_model(x, a, b):
    return (np.exp(a + b*x))/(1 + np.exp(a + b*x)) 

n = np.array([200])
loc_x = np.array([0])
scale_x = np.array([1])
scale_nu = np.array([1]) #np.array([0.000001, 0.5, 1, 2]) # Specify different values of stadnard deviation of ME (try as many as you want)
scale_eps = np.array([0.5])  # True value of \sigma^2_{\epsilon}
B = np.array([500])
m = np.array([1])
c = np.array([100]) # You can also try different values of c parameter in DP
T = np.array([100]) # Truncated limit of DP sum
p_ = np.array([3]) 
theta_star = np.array([1,2])  # True parameter values
R = 100

if __name__=='__main__':
    var_nu = [i**2 for i in scale_nu]
    num_scales = len(scale_nu)
    num_cs = len(c)
    num_config = len(scale_nu)*len(c)
    posterior_samples = np.zeros((int(B[0]), len(theta_star), num_config, R))
    data_sets = np.zeros((int(n[0]), 2, num_config, R))
    x_sets = np.zeros((int(n[0]), num_config, R))
    coefficients = np.zeros((len(theta_star), num_config, R))
    times = []
    for r in tqdm(range(R)):
        print(f'---Replication {r}---')
        args.seed1 += 1
        args.seed2 += 1
        for p,params in enumerate(list(product(n,loc_x,scale_x,scale_nu,scale_eps,B,m,c,T,p_))):
            print(f'---Running configutation {p}---')
            sample, data, x, total = train_npl(np.array(params), reg_func, args.seed1, args.seed2) #mmd_est,
            posterior_samples[:, :, p, r] = sample.reshape((int(B[0]), len(theta_star)))
            data_sets[:, :, p, r] = data
            x_sets[:, p, r] = x
            times.append({
                "replication": r,
                "config_id": p,
                "n": params[0],
                "scale_nu": params[3],
                "c": params[7],
                "B": params[5],
                "time_sec": total
            })
            np.savetxt(folder_path+f'sample_scale_nu{params[3]}_c{params[7]}_n{params[0]}_B{params[5]}_seed1{args.seed1}_seed2{args.seed2}.txt', posterior_samples[:, :, p, r])
    df_times = pd.DataFrame(times)
    df_times.to_csv(folder_path + "runtime_summary.csv", index=False)    
            
            
    