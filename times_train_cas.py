from utils import process_csv
import time
from npl_poly import npl
import numpy as np
from itertools import product
from scipy.optimize import curve_fit
import pandas as pd
from tqdm import tqdm
from timeit import default_timer as timer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int)
args = parser.parse_args()

args.seed
folder_path = '/dcs/pg23/u1604520/mem/results/times/cas/'

def reg_func(theta,x):
    return theta[0] + theta[1]*x + theta[2]*x**2

def train_npl(params, seed):
    B,m,c,T,p_ = params
    data, x = process_csv('./me-models/CASschools.csv')
    npl_ = npl(data, int(B),int(m), c, T, int(p_), seed, lx=-1, ly=-1, prior=np.array([0.5]), me_type='berkson')  #0.2
    start = timer()
    npl_.draw_samples()
    end = timer()
    total = end - start
    sample = npl_.sample
    return sample, data, x, total 

# Define a nonlinear model function 
def nonlinear_model(x, a, b, c):
    return a + b*x + c*x**2

B = np.array([200])
m = np.array([1])
c = np.array([100]) # You can also try different values of c parameter in DP
T = np.array([100]) # Truncated limit of DP sum
p_ = np.array([4]) 
R = 100

if __name__=='__main__':
    num_cs = len(c)
    num_config = len(c)
    posterior_samples = np.zeros((int(B[0]), int(p_[0] - 1), num_config, R))
    coefficients = np.zeros((int(p_[0] - 1), num_config, R))
    times = []
    for r in tqdm(range(R)):
        args.seed += 1
        for p,params in enumerate(list(product(B,m,c,T,p_))):
            sample, data, x, total = train_npl(np.array(params), args.seed)
            posterior_samples[:, :, p, r] = sample.reshape((int(B[0]), int(p_[0] - 1)))
            np.savetxt(folder_path+f'sample_c{params[2]}_B{params[0]}_seed{args.seed}.txt', posterior_samples[:, :, p, r])

            times.append({
                "replication": r,
                "config_id": p,
                "c": params[2],
                "B": params[0],
                "time_sec": total
            })
        
    df_times = pd.DataFrame(times)
    df_times.to_csv(folder_path + "runtime_summary.csv", index=False)         
    