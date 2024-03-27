import numpy as np
from npl import npl_class
import time
from utils import process_csv
from tqdm import tqdm 

folder_path = './results/mh/bcr_sim_data2/'

def train_(params, path_data, r):
    B, m, c, T = params
    #data_control, data_treat = process_csv('./me-models/mental_health_study/mental_health_data.csv')
    data_control, data_treat = process_csv(path_data)
    data = np.vstack((data_control, data_treat))
    npl_ = npl_class(data,int(B),int(m), c, T, r, lx=10, ly=100, prior=1)
    t0 = time.time()
    npl_.draw_samples()
    t1 = time.time()
    total = t1-t0
    sample = npl_.sample
    knots_sample = npl_.knots_sample
    np.savetxt(folder_path+f'bootstrap_samples_bcr_{r}.txt', sample.reshape((int(B),64)))
    np.savetxt(folder_path+f'bootstrap_samples_knots_bcr_{r}.txt', knots_sample.reshape((int(B),30)))
    return sample, knots_sample
      
B = 200 
m = 1
c = 1
T = 100
R = 56

params = np.array([B,m,c,T])

if __name__=='__main__':
    for r in tqdm(range(R)):
        path_data = f'./mh_study/sim_data_bcr/sim_data_bcr_{r+45}.csv'
        sample, knots_sample = train_(params, path_data, r+45)