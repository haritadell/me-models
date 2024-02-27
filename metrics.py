import numpy as np
from scipy.optimize import curve_fit
from utils import sample_observed_data_berkson, sample_observed_data_classical

confidence_level = 0.90
num_realizations = 100
seed = 12
folder_path = '/dcs/pg23/u1604520/mem/results/exponential_classical_unifstart/'

scale_nu = 2.0 #0.000001 #1.0 #2.0
c = 100
n = 200
B = 500
theta_star = np.array([1,2]) #,3
loc_x = 0
scale_x = 1
scale_eps = 0.5
mses = np.zeros((num_realizations, 2, len(theta_star)))
stds = np.zeros((num_realizations, 2, len(theta_star)))
counts = np.zeros((len(theta_star)))

def nonlinear_model(x, a, b):
    return (np.exp(a + b*x))/(1 + np.exp(a + b*x)) #a + b*x + c*x**2

def reg_func(theta,x):
    return (np.exp(theta[0] + theta[1]*x))/(1 + np.exp(theta[0] + theta[1]*x)) #theta[0] + theta[1]*x + theta[2]*x**2 # #


for r in range(num_realizations):
    seed += 1
    boot_sample = np.loadtxt(folder_path+f'sample_scale_nu{scale_nu}_c{c}_n{n}_B{B}_seed{seed}.txt')
    data, x = sample_observed_data_classical(reg_func, int(n), loc_x, scale_x, scale_nu, scale_eps, theta_star, seed)#np.loadtxt(folder_path+f'data_scale_nu{scale_nu}_c{c}_n{n}_B{B}_seed{seed}.txt')
    mean_boot_sample = boot_sample.mean(axis=0)
    initial_guess = [0,4] #[0, 4]  # Initial parameter guess
    ls_estimator, _ = curve_fit(nonlinear_model, data[:,0], data[:,1], p0=initial_guess)
    #print(boot_sample.mean(axis=0))
    #mses[r, :, :] = np.loadtxt(folder_path+f'mses_scale_nu{scale_nu}_c{c}_n{n}_B{B}_seed{seed}.txt')
    mses[r, 0, :] = np.asarray((mean_boot_sample - theta_star)**2)
    mses[r, 1, :] = np.asarray((ls_estimator - theta_star)**2)

    # Credible interval 
    for i in range(len(theta_star)):
        count = 0
        lower_bound = np.percentile(boot_sample[:, i], (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(boot_sample[:, i], (1 + confidence_level) / 2 * 100)
        if lower_bound <= theta_star[i] <= upper_bound:
            count += 1
        counts[i] += count
      
# Calculate the coverage probability
coverage_probabilities = counts / num_realizations
mses_over_runs = np.mean(mses, axis=0)
stds_mses_over_runs = np.std(mses, axis=0)
print(f"Coverage Probability for theta_1 for ME std {scale_nu}: {coverage_probabilities[0] * 100}%")
print(f"Coverage Probability for theta_2 for ME std {scale_nu}: {coverage_probabilities[1] * 100}%")
#print(f"Coverage Probability for theta_3 for ME std {scale_nu}: {coverage_probabilities[2] * 100}%")
print(f"Mean Squared error for Robust-MEM: {mses_over_runs[0, :]}")
print(f"Mean Squared error for Least Squares: {mses_over_runs[1, :]}")
print(f"Std - Mean Squared error for Robust-MEM: {stds_mses_over_runs[0, :]}")
print(f"Std - Mean Squared error for Least Squares: {stds_mses_over_runs[1, :]}")