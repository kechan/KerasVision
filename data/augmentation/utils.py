import numpy as np

# Utilities for performing rotation on "y" with (x, y, r), a square

def rot0(y):    #identity
  return y

def rot90(y):
  out_y = np.copy(y)
  
  # left/right flip
  out_y[1] = 1. - out_y[1]

  # transpose
  tmp = out_y[1]
  out_y[1] = out_y[2]
  out_y[2] = tmp
  
  return out_y

def rot180(y):
  out_y = np.copy(y)
  
  # left/right flip 
  out_y[1] = 1. - out_y[1]
  
  # up/down flip 
  out_y[2] = 1. - out_y[2]
  
  return out_y
  
def rot270(y):
  out_y = np.copy(y)
  
  # up-down flip
  out_y[2] = 1. - out_y[2]
  
  # transpose
  tmp = out_y[1]
  out_y[1] = out_y[2]
  out_y[2] = tmp
  
  return out_y

def rot_func(k):
  if k == 1:
    return rot90
  elif k == 2:
    return rot180
  elif k == 3:
    return rot270
  else:
    return rot0
