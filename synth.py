import os
from time import time
from glob import glob

import numpy as np
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from util import files,geom,num,util

def synth(di):
  """
  Collect all of the metrics files for anylisis
  Returns X, j
      where X is a N X M matrix where N is number of runs and M is number of metrics
      where j is the labels for M
  """
  dfis = glob( os.path.join(di, '*'+"_metrics.py") )

  data_array = []
  _, j_flat = get_metrics(dfis[0])
  for dfi in dfis:
    data, names = get_metrics(dfi)
    assert names == j_flat
    data_array.append(data)
  return np.vstack(data_array), {k:i for i,k in enumerate(j_flat)}

def get_metrics(filename):
  """
  Grab contents of metrics from each file
  returns 1XM matrix
  """
  dict_str = open(filename).read()
  data_dict = eval(dict_str)
  print data_dict.keys()
  flat = [data_dict[x] for x in data_dict.keys()]
  return flat, data_dict.keys()
