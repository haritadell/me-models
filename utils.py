import numpy as np
import jax.numpy as jnp
from jax import vmap
import scipy.stats as stats

def sample_observed_data_berkson(reg_func, n,loc_x, scale_x, scale_nu, scale_eps, theta, seed):
    """Function to sample observations with Berkson measurement error."""

    # You can either sample equidistant points or from a normal distribution
    # TODO: Sample multiple values of (x,y) for each w (group)
    # w = np.linspace(0,5,num=int(n/5))
    # w = np.repeat(w, 5)
    w = stats.norm.rvs(loc=loc_x, scale=scale_x, size=(n,), random_state=seed+n+2) # covariates with Berkson ME
    nu = stats.norm.rvs(loc=0, scale=scale_nu, size=(n,), random_state=seed+n) # ME
    x = w + nu # True covariates
    eps = stats.norm.rvs(loc=0, scale=scale_eps, size=(n,), random_state = seed+n+3) # Outcome variable (Y) error
    #Y = theta[0] + theta[1]*x + theta[2]*x**2 + theta[3]*x**3 + eps    #theta[0] +
    Y = reg_func(theta,x) + eps
    X_train = w
    Y_train = Y
    data = np.zeros((n,2))
    data[:,0] = X_train
    data[:,1] = Y_train
    # return observed data (with ME) and the true covariates x
    return data, x

def sqeuclidean_distance(x, y):
    return jnp.sum((x-y)**2)

def rbf_kernel(x1, y1, x2, y2, lx, ly):
    KX = jnp.exp( -(1/(2*lx**2)) * sqeuclidean_distance(x1, x2))
    KY = jnp.exp( -(1/(2*ly**2)) * sqeuclidean_distance(y1, y2))
    return KX*KY

def k_jax(x1,y1,x2,y2,lx,ly):
    """Gaussian kernel compatible with JAX library"""

    mapx1 = vmap(lambda x1, y1, x2, y2: rbf_kernel(x1, y1, x2, y2, lx, ly), in_axes=(0, 0, None, None), out_axes=0)
    mapx2 = vmap(lambda x1, y1, x2, y2: mapx1(x1, y1, x2, y2), in_axes=(None, None, 0, 0), out_axes=1)
    K = mapx2(x1, y1, x2, y2)

    return K

def mse(theta,theta_star):
    se = np.asarray((theta-theta_star))**2
    mse_ = np.mean(se,axis=1)
    std_ = np.std(se,axis=1)

    return mse_, std_ 