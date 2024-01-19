import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import vmap

def process_csv(path_to_csv_file):
  df = pd.read_csv(path_to_csv_file)
  df_control = df.loc[df['status'] == 'control']
  df_treat = df.loc[df['status'] == 'treatment']
  # create control matrix
  W_c = np.array([df_control["w"]]).flatten()
  Y_c = np.array([df_control["y"]]).flatten()
  n = len(W_c)
  data_control = np.zeros((n,2))
  data_control[:,0] = W_c
  data_control[:,1] = Y_c
  # create treatment matrix 
  W_t = np.array([df_treat["w"]]).flatten()
  Y_t = np.array([df_treat["y"]]).flatten()
  n = len(W_t)
  data_treat = np.zeros((n,2))
  data_treat[:,0] = W_t
  data_treat[:,1] = Y_t
  return data_control, data_treat

def make_design_matrices(xsample,Z):
  # Z is (n,K)
  n = len(xsample)
  # make X
  X = jnp.ones((n,4))
  X = X.at[:,1].set(xsample)
  I = jnp.zeros(n)
  I = I.at[:245].set(1)
  X = X.at[:,2].set(1-I)
  X = X.at[:,3].set(jnp.multiply((1-I),xsample))
  # make Z_0, Z_1
  K = Z.shape[1]
  Z_0 = jnp.ones((n,K))
  Z_1 = jnp.ones((n,K))
  def make_Z(I,Zk):
    return jnp.multiply(I,Zk)

  Z_0 = vmap(make_Z, in_axes=(None,1), out_axes=1)(I,Z)
  Z_1 = vmap(make_Z, in_axes=(None,1), out_axes=1)(1-I,Z)

  return X, Z_0, Z_1

def create_knots(xsample,K,k):
    return ((K+1-k)*jnp.min(xsample)+k*jnp.max(xsample))/(K+1)

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

def k_comp(x1,y1,x2,y2):
    l_range = np.array([0.1,1.0,10.0,100.0]) 

    k1 = vmap(k_jax, in_axes=(None,None,None,None,0,None)) #
    k2 = vmap(k1, in_axes=(None,None,None,None,None,0))(x1,y1,x2,y2,l_range,l_range)
    k3 = jnp.sum(jnp.sum(k2,axis=1), axis=0) 
 
    return k3