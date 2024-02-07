from jax import vmap, lax, random
import jax.numpy as jnp
from utils import create_knots, process_csv
import numpy as np
from npl import npl_class

seed = 17
folder_path = './results/'

def simulate_data(theta, x, knots, rng):
    #var_eps = jnp.exp(theta[-1])   # variance of error on y
    var_eps = 0.5
    u = theta #[:-1]
    N = len(x)
    K = 10
    def fn_to_scan(B, j):
        cond = (x>=knots[j])&(x<=knots[j+1])
        act_x = jnp.where(cond,x,0)
        resc_x = (act_x-knots[j])/(knots[j+1]-knots[j])
        B = B.at[:,j].set(jnp.where(cond,(1/2)*(1-resc_x)**2,B[:,j]))
        B = B.at[:,j+1].set(jnp.where(cond,-(resc_x**2)+resc_x+1/2,B[:,j+1]))
        B = B.at[:,j+2].set(jnp.where(cond,(resc_x**2)/2,B[:,j+2]))

        return B, j+1

    Z = lax.scan(fn_to_scan, jnp.zeros((N,K+1)), jnp.arange(K-1))[0]
    #X = make_design_matrix(xsample) # X is Nx3, Zs are NxK (22)
    y = random.multivariate_normal(rng, jnp.matmul(Z,u) , var_eps*jnp.eye(N)) 
    
    return y

if __name__=='__main__':
    theta_sarkar = np.loadtxt('thetas_nhanes.txt')
    x_sarkar = np.loadtxt('xs_nhanes.txt')
    data = process_csv('Data_similar_to_Sodium_from_EATS.csv')
    start_xs = data[:,0]
    range_start_xs = max(start_xs) - min(start_xs)
    range_expand = 0.01
    min_knots = min(-3,min(start_xs)-range_expand*range_start_xs)
    max_knots = max(3,max(start_xs)+range_expand*range_start_xs)
    knots = create_knots(min_knots, max_knots, 10) # create knots as from Sarkar et al.
    # key = random.PRNGKey(13)
    # y_sarkar = simulate_data(theta_sarkar, x_sarkar, knots, key)
    # data_sarkar = np.zeros((len(y_sarkar), 2))
    # data_sarkar[:, 0] = data[:,0]
    # data_sarkar[:, 1] = y_sarkar
    #np.savetxt('data_sarkar.txt', data_sarkar)
    data_sarkar = np.loadtxt('data_sarkar.txt')
    B = 200 
    m = 1
    c = 1
    T = 100     
    npl_ = npl_class(data_sarkar,B,m, c, T, seed, knots, lx=100, ly=10, prior=1.0)
    npl_.draw_samples()
    sample = npl_.sample
    np.savetxt(folder_path+f'bootstrap_samples_npl.txt', sample.reshape((int(B),12)))
    
    