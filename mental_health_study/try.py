import numpy as np
from utils import process_csv

folder_path = './results/'
theta_npl = np.loadtxt(folder_path+'bootstrap_samples.txt').mean(axis = 0)
knots_npl = np.loadtxt(folder_path+'bootstrap_samples_knots.txt').mean(axis = 0)

def ate(x, theta, knots):
    c = np.zeros(len(x))
    beta_0_drug = theta[2]
    beta_1_drug = theta[3]
    u_0 = theta[4:34]
    u_1 = theta[34:]
    for j in range(len(x)):
        c[j] = beta_0_drug + beta_1_drug*x[j]
        for i in range(len(knots)):
            if x[j] > knots[i]:
                z_i = x[j] - knots[i]
            else:
                z_i = 0
            c[j] += (u_1[i] - u_0[i])*z_i
    return c

data_control, data_treat = process_csv('sim_data_npl.csv')
data = np.vstack((data_control, data_treat))
x = np.loadtxt('x_npl.txt')
reg_true = ate(x, theta_npl, knots_npl)



    