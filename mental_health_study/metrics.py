import numpy as np
import matplotlib.pyplot as plt
import copy


confidence_level = 0.9
num_realizations = 1
seed = 17
folder_path = './results/'

c = 100
n = 200
B = 200
#theta_star = np.loadtxt(folder_path+'bootstrap_samples.txt').mean(axis = 0)
#knots =  np.loadtxt(folder_path+'bootstrap_samples_knots.txt').mean(axis = 0)
theta_star = np.zeros(64)
theta_star[:4] = np.loadtxt('bcr_beta_star.txt')
theta_star[4:35] = np.loadtxt('bcr_u_control_star.txt')
theta_star[35:] = np.loadtxt('bcr_u_treatment_star.txt')

mses = np.zeros((num_realizations, 2, len(theta_star)))
stds = np.zeros((num_realizations, 2, len(theta_star)))
count = 0

def ate(x, theta, knots):
    c = np.zeros(len(x))
    beta_0_drug = theta[2]
    print(beta_0_drug)
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

for r in range(num_realizations):
    seed += 1
    
    bcr_beta = np.loadtxt('bcr_beta_npl.txt').transpose()
    bcr_ucontrol = np.loadtxt('bcr_ucontrol_npl.txt').transpose()
    bcr_utrmt = np.loadtxt('bcr_utreatment_npl.txt').transpose()
    bcr_knots = np.loadtxt('bcr_knots_npl.txt').transpose().mean(axis=0)
    
    bcr_sample = np.zeros((1000, len(theta_star)))
    bcr_sample[:, :4] = bcr_beta[:, :4]
    bcr_sample[:, 1] = bcr_beta[:, 2]
    bcr_sample[:, 2] = bcr_beta[:, 1]
    bcr_sample[:, 4:34] = bcr_ucontrol
    bcr_sample[:, 34:] = bcr_utrmt
    mean_bcr_sample = bcr_sample.mean(axis=0)
    
    boot_sample = np.loadtxt(folder_path+'bootstrap_samples_npl.txt')
    mean_boot_sample = boot_sample.mean(axis=0)
    #mean_boot_sample[:4] = mean_bcr_sample[:4].copy()
    x = np.loadtxt('x_npl.txt')
    knots_npl = np.loadtxt(folder_path+'bootstrap_samples_knots_npl.txt').mean(axis=0)
    ate_npl = ate(x, mean_boot_sample, knots)
    
    ate_bcr = ate(x, mean_bcr_sample, knots)
    ate_true = ate(x, theta_star, knots)
    
    mses[r, 0, :] = np.asarray((mean_boot_sample - theta_star)**2)
    mses[r, 1, :] = np.asarray((mean_bcr_sample - theta_star)**2)

    # Credible interval
    lower_bound = np.percentile(boot_sample[:, 1], (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(boot_sample[:, 1], (1 + confidence_level) / 2 * 100)
    #print(lower_bound, upper_bound)

    if lower_bound <= theta_star[1] <= upper_bound:
        count += 1
      
# Calculate the coverage probability
coverage_probability = count / num_realizations
mses_over_runs = np.mean(mses, axis=0)
# print(f"Coverage Probability: {coverage_probability * 100}%")
# print(f"Mean Squared error for Robust-MEM: {mses_over_runs[0, :]}")
# print(f"Mean Squared error for BCR: {mses_over_runs[1, :]}")
# print(np.mean(mses_over_runs, axis=1))
# print(np.std(mses_over_runs, axis=1)**2)

plt.scatter(range(len(theta_star)), mses_over_runs[0, :], marker='x', label='RobustMEM (MMD)')
plt.scatter(range(len(theta_star)), mses_over_runs[1 :], marker='o', label='BCR')
plt.legend()
plt.savefig('mses_plot.png')

# plt.scatter(x, ate_npl, label='npl')
# plt.scatter(x, ate_bcr, label='bcr')
# plt.scatter(x, ate_true, label='true')
# plt.legend()
# plt.savefig('ates_plot.png')

# p = len(theta_star)
# #plt.scatter(range(p), theta_star, label='true')
# plt.scatter(range(p), abs(mean_boot_sample-theta_star), label='npl')
# plt.scatter(range(p), abs(mean_bcr_sample-theta_star), label='bcr')
# plt.legend()
# plt.savefig('params_plot.png')

