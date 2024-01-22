import numpy as np
from npl import npl_class
import time
from utils import process_csv, create_knots

seed = 17
folder_path = './results/'

def train_(params):
    B, m, c, T = params
    data = process_csv('Data_similar_to_Sodium_from_EATS.csv')
    start_xs = data[:,0]
    range_start_xs = max(start_xs) - min(start_xs)
    range_expand = 0.01
    min_knots = min(-3,min(start_xs)-range_expand*range_start_xs)
    max_knots = max(3,max(start_xs)+range_expand*range_start_xs)
    # knots = create_knots(-3,12.74465,10) # create knots as from Sarkar et al.
    knots = create_knots(min_knots, max_knots, 10) # create knots as from Sarkar et al.
    npl_ = npl_class(data,int(B),int(m), c, T, seed, knots, lx=10, ly=100, prior=1.0)
    t0 = time.time()
    npl_.draw_samples()
    t1 = time.time()
    total = t1-t0
    sample = npl_.sample
    np.savetxt(folder_path+f'bootstrap_samples.txt', sample.reshape((int(B),11)))
    return sample
      
B = 200 
m = 1
c = 50
T = 100

params = np.array([B,m,c,T])

if __name__=='__main__':
    sample = train_(params)