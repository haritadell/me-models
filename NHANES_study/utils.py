import numpy as np
import pandas as pd
import jax.numpy as jnp

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
