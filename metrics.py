import numpy as np
from scipy.optimize import curve_fit

confidence_level = 0.95
num_realizations = 100
seed = 12
folder_path = './results/exponential_classical_unifstart/'

scale_nu = 0.000001 #1.0 #2.0
c = 100
n = 200
B = 200
theta_star = np.array([1,2])

mses = np.zeros((num_realizations, 2, len(theta_star)))
stds = np.zeros((num_realizations, 2, len(theta_star)))
count = 0

def nonlinear_model(x, a, b):
    return (np.exp(a + b*x))/(1 + np.exp(a + b*x))

for r in range(num_realizations):
    seed += 1
    boot_sample = np.loadtxt(folder_path+f'sample_scale_nu{scale_nu}_c{c}_n{n}_B{B}_seed{seed}.txt')
    data = np.loadtxt(folder_path+f'data_scale_nu{scale_nu}_c{c}_n{n}_B{B}_seed{seed}.txt')
    mean_boot_sample = boot_sample.mean(axis=0)
    initial_guess = [0, 4]  # Initial parameter guess
    ls_estimator, _ = curve_fit(nonlinear_model, data[:,0], data[:,1], p0=initial_guess)
    #print(boot_sample.mean(axis=0))
    #mses[r, :, :] = np.loadtxt(folder_path+f'mses_scale_nu{scale_nu}_c{c}_n{n}_B{B}_seed{seed}.txt')
    mses[r, 0, :] = np.asarray((mean_boot_sample - theta_star)**2)
    mses[r, 1, :] = np.asarray((ls_estimator - theta_star)**2)

    # Credible interval
    lower_bound = np.percentile(boot_sample[:, 1], (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(boot_sample[:, 1], (1 + confidence_level) / 2 * 100)
    #print(lower_bound, upper_bound)

    if lower_bound <= theta_star[1] <= upper_bound:
        count += 1
      
# Calculate the coverage probability
coverage_probability = count / num_realizations
mses_over_runs = np.mean(mses, axis=0)
print(f"Coverage Probability for ME std {scale_nu}: {coverage_probability * 100}%")
print(f"Mean Squared error for Robust-MEM: {mses_over_runs[0, :]}")
print(f"Mean Squared error for Least Squares: {mses_over_runs[1, :]}")