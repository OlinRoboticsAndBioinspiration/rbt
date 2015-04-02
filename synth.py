import os
from time import time
from glob import glob
import rbt

import numpy as np
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from util import files,geom,num,util

def synth(di, take_valid=True):
  """
  Collect all of the metrics files for anylisis
  Returns X, j
      where X is a N X M matrix where N is number of runs and M is number of metrics
      where j is the labels for M
  """
  #TODO move configuration (metrics dir name) to global config
  di = os.path.join(di, "metrics")
  dfis = glob( os.path.join(di, '*'+"_metrics.py") )
  if len(dfis) == 0:
      print "Warning: No files found"
      return np.array([]), {}, []

  data_array = []
  _, j_flat = get_metrics(dfis[0])
  for dfi in dfis:
    data, names = get_metrics(dfi)
    assert names == j_flat
    data_array.append(data)
  base_files = [''.join(x.split("_")[0:-1]) for x in dfis]
  data = np.vstack(data_array)
  keys = {k:i for i,k in enumerate(j_flat)}
  valid_entries = data[..., keys["is_valid"]] == 1
  base_files = [base_files[i] for i, e in enumerate(data[..., keys["is_valid"]]) if e == True]
  print -np.sum(valid_entries-1), "bad trials"
  data = data[valid_entries, ...]
  return data, keys, base_files

def get_metrics(filename):
  """
  Grab contents of metrics from each file
  returns 1XM matrix
  """
  dict_str = open(filename).read()
  data_dict = eval(dict_str)
  flat = [data_dict[x] for x in data_dict.keys()]
  return flat, data_dict.keys()
