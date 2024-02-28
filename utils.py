import numpy as np
import jax.numpy as jnp
from jax import vmap
import jax.numpy.linalg as la
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import scipy.odr as odr

def sample_observed_data_berkson(reg_func, n,loc_x, scale_x, scale_nu, scale_eps, theta, seed):
    """Function to sample observations with Berkson measurement error."""

    # You can either sample equidistant points or from a normal distribution
    # TODO: Sample multiple values of (x,y) for each w (group)
    # w = np.linspace(0,5,num=int(n/5))
    # w = np.repeat(w, 5)
    gen = np.random.default_rng(seed=seed)
    w = stats.norm.rvs(loc=loc_x, scale=scale_x, size=(n,), random_state=gen) # covariates with Berkson ME
    nu = stats.norm.rvs(loc=0, scale=scale_nu, size=(n,), random_state=gen) # ME
    x = w + nu # True covariates
    eps = stats.norm.rvs(loc=0, scale=scale_eps, size=(n,), random_state=gen) # Outcome variable (Y) error
    #Y = theta[0] + theta[1]*x + theta[2]*x**2 + theta[3]*x**3 + eps    #theta[0] +
    Y = reg_func(theta,x) + eps
    X_train = w
    Y_train = Y
    data = np.zeros((n,2))
    data[:,0] = X_train
    data[:,1] = Y_train
    # return observed data (with ME) and the true covariates x
    return data, x

def sample_observed_data_classical(reg_func, n,loc_x, scale_x, scale_nu, scale_eps, theta, seed):
    """Function to sample observations with Berkson measurement error."""
    gen = np.random.default_rng(seed=seed)
    x = stats.norm.rvs(loc=loc_x, scale=scale_x, size=(n,), random_state=gen) 
    nu = stats.norm.rvs(loc=0, scale=scale_nu, size=(n,), random_state=gen) # ME
    w = x + nu # noisy observed covariates
    eps = stats.norm.rvs(loc=0, scale=scale_eps, size=(n,), random_state = gen) # Outcome variable (Y) error
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
    mse_ = np.mean(se,axis=0)
    std_ = np.std(se,axis=0)

    return mse_, std_ 

def tls(X,y):
    X_ = X - jnp.mean(X)
    y_ = y - jnp.mean(y)
    if X_.ndim == 1: 
        d=1
        X_ = X_.reshape((len(X_),d))
    else:
        d = jnp.array(X_).shape[1]
    
    Z = jnp.vstack((X_.T,y_)).T #Z is (n,2)
    U, s, Vt = la.svd(Z, full_matrices=True)
    
    V = Vt.T
    Vxy = V[:d,d:]
    Vyy = V[d:,d:]
    a_tls = - Vxy  / Vyy # total least squares soln
    b_tls = jnp.mean(y) - a_tls*jnp.mean(X)
    
    return jnp.array([b_tls[0], a_tls[0]]) 


def run_odr(data):
  lr = LinearRegression(fit_intercept=True)
  lr.fit(data[:,0].reshape(-1, 1), data[:,1].reshape(-1, 1))
  ww = lr.coef_
  bb = lr.intercept_
  #ols_params = np.array([ww[0][0],bb[0]]) 
  def f(B, x):
    return B[0] + B[1]*(x)
  linear = odr.Model(f)
  mydata = odr.Data(data[:,0], data[:,1])
  myodr = odr.ODR(mydata, linear, beta0=[bb[0], ww[0][0]]) 
  myoutput = myodr.run()
  return myoutput

def run_ols(data):
  lr = LinearRegression(fit_intercept=True)
  lr.fit(data[:,0].reshape(-1, 1), data[:,1].reshape(-1, 1))
  ww = lr.coef_
  bb = lr.intercept_
  ols_params = np.array([ww[0][0],bb[0]])
  return ols_params

