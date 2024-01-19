import numpy as np
from npl import npl_class
import time
from utils import process_csv

seed = 17
folder_path = './results/'

def train_(params):
    B, m, c, T = params
    data = process_csv('nhanes_data.csv')
    npl_ = npl_class(data,int(B),int(m), c, T, seed, lx=10, ly=100, prior=1.0)
    t0 = time.time()
    npl_.draw_samples()
    t1 = time.time()
    total = t1-t0
    sample = npl_.sample
    knots_sample = npl_.knots_sample
    np.savetxt(folder_path+f'bootstrap_samples.txt', sample.reshape((int(B),64)))
    np.savetxt(folder_path+f'bootstrap_samples_knots.txt', knots_sample.reshape((int(B),30)))
    return sample, knots_sample
      
B = 200 
m = 1
c = 50
T = 100

params = np.array([B,m,c,T])

if __name__=='__main__':
    sample, knots_sample = train_(params)