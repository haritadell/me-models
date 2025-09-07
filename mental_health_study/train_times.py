import numpy as np
from npl import npl_class
from utils import process_csv
from tqdm import tqdm 
from timeit import default_timer as timer
import pandas as pd

folder_path = '/dcs/pg23/u1604520/mem/results/times/mh/'

def train_(params, path_data, r):
    B, m, c, T = params
    #data_control, data_treat = process_csv('./me-models/mental_health_study/mental_health_data.csv')
    data_control, data_treat = process_csv(path_data)
    data = np.vstack((data_control, data_treat))
    npl_ = npl_class(data,int(B),int(m), c, T, r, lx=10, ly=100, prior=1)
    start = timer()
    npl_.draw_samples()
    end = timer()
    total = end - start 
    sample = npl_.sample
    knots_sample = npl_.knots_sample
    np.savetxt(folder_path+f'bootstrap_samples_bcr_{r}.txt', sample.reshape((int(B),64)))
    np.savetxt(folder_path+f'bootstrap_samples_knots_bcr_{r}.txt', knots_sample.reshape((int(B),30)))
    return sample, knots_sample, total
      
B = 200 
m = 1
c = 1
T = 100
R = 10

params = np.array([B,m,c,T])

if __name__=='__main__':
    times = []
    for r in tqdm(range(R)):
        path_data = f'/dcs/pg23/u1604520/mem/sim_data_bcr/sim_data_bcr_{r+1}.csv'
        sample, knots_sample, total = train_(params, path_data, r)
        times.append({
            "replication": r,
            "time_sec": total
        })
    df_times = pd.DataFrame(times)
    df_times.to_csv(folder_path + "runtime_summary.csv", index=False)