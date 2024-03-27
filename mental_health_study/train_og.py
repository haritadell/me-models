'''Fit Robust-MEM (MMD) for the mental health study using real data'''

import numpy as np
from npl import npl_class
import time
from utils import process_csv
from tqdm import tqdm 

folder_path = './results/mh/original_data/'

def train_(params):
    seed = 13
    B, m, c, T = params
    data_control, data_treat = process_csv('./me-models/mental_health_study/mental_health_data.csv')
    data = np.vstack((data_control, data_treat))
    npl_ = npl_class(data,int(B),int(m), c, seed, T, lx=10, ly=100, prior=1)
    t0 = time.time()
    npl_.draw_samples()
    t1 = time.time()
    total = t1-t0
    sample = npl_.sample
    knots_sample = npl_.knots_sample
    np.savetxt(folder_path+f'bootstrap_samples_bcr.txt', sample.reshape((int(B),64)))
    np.savetxt(folder_path+f'bootstrap_samples_knots_bcr.txt', knots_sample.reshape((int(B),30)))
    return sample, knots_sample
      
B = 200 
m = 1
c = 1
T = 100

params = np.array([B,m,c,T])

if __name__=='__main__':
    sample, knots_sample = train_(params)