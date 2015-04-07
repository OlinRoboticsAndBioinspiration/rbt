
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Sam Burden, UC Berkeley, 2013 
# Modified by Luke Metz, Olin, 2014

import os
import sys
from time import time
from glob import glob
from rbtFunctions import load, crop, mcu, circle_fit, cal
from rbtFunctions import dat, geom, json_task, ukf, plt, plane
from rbtFunctions import vis3d, cfg_metrics, metrics, line_angle
from rbtFunctions import time_metric, standard_transform, quality

import numpy as np
import numpy.linalg as la
from scipy import optimize
from scipy import stats

from joblib import Parallel, delayed

air = ['dat','plt']
cal_task = ['dat','plane','cal','plt']
ukf_task = ['dat','cal','geom','ukf', 'json_task']#,'plt']
plot = ['load', 'plt']
skel = ['dat','geom','plt']
fitness = ['load','crop', 'mcu', 'circle_fit', 'cfg_metrics', 'time_metric', 'metrics']

dsfx = {'opti':'_mocap.csv','vicon':'.dcr','phase':'.txt'}
ddir = 'dat'

def do(di,dev=None,trk='rbt',procs=ukf,exclude='_ukf.npz', n_jobs=1, **kwds):
  """
  Process all unprocessed rigid body data

  Inputs:
    di - str - directory containing data
    dev - str - motion capture hardware
    trk - str - name of rigid body trackable
    procs - [str,...] - processes to do

  Outputs:
    rbs - list of rigid body structs

  See also:
    do_() is called on each file
  """
  good_files = get_runs(di, dev, exclude)
  print good_files, "<<good files"
  if (n_jobs != 1):
      rbs = Parallel(n_jobs=n_jobs)(
          delayed(do_)(f, dev=dev, trk=trk, procs=procs, **kwds) for f in good_files)
  else:
      rbs =[do_(f, dev=dev, trk=trk, procs=procs, **kwds) for f in good_files]
  print "Finished running through files"
  return rbs

def get_runs(di, dev=None, exclude="_exclude"):
  sfx = dsfx[dev]
  dfis = glob( os.path.join(di, '*'+sfx) )
  efis = glob( os.path.join(di, ddir, '*'+exclude) )
  good_files= []
  for dfi in dfis:
    _,fi = os.path.split(dfi)
    fi = fi.split(sfx)[0]
    if dev == 'phase' or '_' not in fi:
      if os.path.join(di, ddir, fi+exclude) not in efis:
        good_files.append(os.path.join(di, fi))
  return good_files


def do_(fi='',dev=None,trk='rbt',procs=[],**kwds):
  """
  Process mocap data, run ukf, generate plots from rigid body data 

  Inputs:
    fi - str - rigid body data file name
    (optional)
    trk - str - name of rigid body trackable
    dev - str - motion capture hardware
    procs - [str,...] - processes to do

  Outputs:
    rb - rigid body struct

  Workflow:
    >> # process mocap data, run ukf, & plot results
    >> rb = Rbt('test/20120612-0910',dev='opti',trk='rbt')
    >> dat.dat(rb)
    >> cal.cal(rb)
    >> geom.geom(rb)
    >> ukf.ukf(rb)
    >> plt.plt(rb)
  """
  print procs
  rb = Rbt(fi,trk=trk,dev=dev)
  for proc in procs:
    cmd = proc + "." + proc + "(rb, **kwds)"
    eval( cmd )
  return rb

class Rbt():
  """
  Rigid body data class

  Workflow:
    >> # process mocap data, run ukf, & plot results
    >> rb = rbt.Rbt('test/20120612-0910')
    >> dat.dat(rb)
    >> ukf.ukf(rb)
    >> plt.plt(rb)

    >> # load processed data & plot results
    >> rb = rbt.Rbt('test/20120612-0910')
    >> load.load(rb)
    >> plt.plt(rb)
  """

  def __init__(self,fi='',trk='rbt',dev=None):
    """
    Rigid body data

    Inputs
      fi - str - mocap data file name
      (optional)
      trk - str - name of rigid body trackable
      dev - str - motion capture hardware
    """
    # TODO fix this properly
    #if '.' in fi:
      #fi,_ = fi.split('.')

    self.fi = fi
    self.t = None #Time series

    ##Dictionary with locations for X.
    #Example: dict(pitch=0,roll=1,yaw=2,x=3,y=4,z=5)
    self.j = None  #Labels for UKF
    self.d = None #Data for each marker
    self.g = None
    self.X = None #UKF data corresponding to self.j
    self.hz = None

    #for crop
    self.start_trial = None
    self.stop_trial = None

    #for mcu
    self.mcu_data = None #MCU data
    self.mcu_j = None #MCU labels

    #for metrics
    self.is_valid = True
    self.metrics_data = {}

    self.trk = trk
    self.dev = dev

if __name__ == '__main__':
  rb = Rbt(sys.argv[1], dev='opti')
  dat.dat(rb)
  cal.cal(rb)
  geom.geom(rb)
  ukf.ukf(rb, viz=100,ord=2)
  load.load(rb)
  crop.crop(rb)
  mcu.mcu(rb)
  cfg_metrics.cfg_metrics(rb)
  metrics.metrics(rb)
  plt.plt(rb)


