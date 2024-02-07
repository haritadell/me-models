from jax import vmap, lax, random
import jax.numpy as jnp
from utils import make_design_matrices, process_csv
import numpy as np
from npl import npl_class

seed = 17
folder_path = './results/'

def simulate_data(theta, x, knots, rng):
    var_eps = 0.5 #jnp.exp(theta[-1])   # variance of error on y
    beta = theta[:4]
    u_0 = theta[4:34]
    u_1 = theta[34:]
        
    N = len(x)

    def create_z(xi,knotsk):
        pred = xi > knotsk
        res = lax.select(pred,1,0)
        return (xi-knotsk)*res
    
    map1 = vmap(create_z, in_axes=(0,None), out_axes=0)
    map2 = vmap(map1, in_axes=(None,0), out_axes=1)
    z = map2(x,knots)

    X, Z_0, Z_1 = make_design_matrices(x,z) # X is Nx3, Zs are NxK (22)

    y = random.multivariate_normal(rng, jnp.matmul(X,beta)+jnp.matmul(Z_0,u_0)+jnp.matmul(Z_1,u_1) , var_eps*jnp.eye(N))
    
    return y

if __name__=='__main__':
    theta_npl = np.loadtxt('./results/bootstrap_samples.txt').mean(axis=0)
    print(theta_npl)
    knots_npl = np.loadtxt('./results/bootstrap_samples_knots.txt').mean(axis=0)
    data_control, data_treat = process_csv('mental_health_data.csv')
    data = np.vstack((data_control, data_treat))
    key = random.PRNGKey(13)
    y_npl = simulate_data(theta_npl, data[:,0], knots_npl, key)
    data_npl = np.zeros((len(y_npl), 2))
    data_npl[:, 0] = data[:,0]
    data_npl[:, 1] = y_npl
    B = 200 
    m = 1
    c = 1
    T = 100     
    npl_ = npl_class(data_npl,B,m, c, T, seed, lx=10, ly=100, prior=1.0)
    npl_.draw_samples()
    sample = npl_.sample
    knots_sample = npl_.knots_sample
    np.savetxt(folder_path+f'bootstrap_samples_npl.txt', sample.reshape((int(B),64)))
    np.savetxt(folder_path+f'bootstrap_samples_knots_npl.txt', knots_sample.reshape((int(B),30)))
    
    
    

