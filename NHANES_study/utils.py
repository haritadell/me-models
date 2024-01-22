import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import vmap 

def process_csv(path_to_csv_file): # data is already scaled
  df = pd.read_csv(path_to_csv_file)
  df = df.groupby("inds").mean()
  # create matrix 
  W = np.array([df["ws"]]).flatten()
  Y = np.array([df["ys"]]).flatten()
  n = len(W)
  data = np.zeros((n,2))
  data[:,0] = W
  data[:,1] = Y
  return data

def make_design_matrix(xsample):
  # Z is (n,K)
  n = len(xsample)
  # make X
  X = jnp.ones((n,2))
  X = X.at[:,1].set(xsample)
  return X

def create_knots(mink, maxk, K):
  knots = jnp.linspace(start=mink, stop=maxk, num=K)
  return knots

def Bbasis(x,knots):
  n = len(x) 
  K = len(knots)
  B = jnp.zeros((n,K+1))
  for j in range(K-1):
    cond = (x>=knots[j])&(x<=knots[j+1]) # boolean for x[i] being between knots[j] and knots[j+1]
    inds = jnp.where(cond) # take those indices that correspond to the above statement being true
    act_x = x[inds] # take the values in x array that correspond to inds
    resc_x = (act_x-knots[j])/(knots[j+1]-knots[j]) # calculate resc_x (same size as act_x)
    B = B.at[inds,j].set((1/2)*(1-resc_x)**2)       # update the jth, j+1th and j+2th columns for rows in inds
    B = B.at[inds,j+1].set(-(resc_x**2)+resc_x+1/2)
    B = B.at[inds,j+2].set((resc_x**2)/2)
  return B

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
