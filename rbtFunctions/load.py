import os
import numpy as np
from util import files,geom,num,util

ddir = 'dat'

def load(rbt,fi=None,dbg=True):
  """
  Load processed data from file

  Inputs:
    fi - str - processed data file, *_dat.npz
  Effects: Loads ukf data to instance
  """
  if (not fi) and (rbt.fi):
    fi = rbt.fi
  if '_' in fi:
    fi,_ = fi.split('_')
  di,fi = os.path.split(fi)
  rbt.fi = os.path.join(di,fi)

  npf = os.path.join(di,ddir,fi+'_dat.npz')
  if os.path.exists(npf):
    if dbg:
      print 'loading '+npf
    npz = np.load( npf ) 
    t=npz['t']; d=npz['d']; hz=npz['hz'];
    rbt.t=t; rbt.d=d; rbt.hz=hz;
    s = util.Struct()
    s.read( os.path.join(di,ddir,fi+'_dat.py'), locals={'array':np.array})
    rbt.trk = s.trk; rbt.dev = s.dev

  npf = os.path.join(di,ddir,fi+'_geom.npz')
  if os.path.exists(npf):
    if dbg:
      print 'loading '+npf
    npz = np.load( npf ) 
    g=npz['g']; pd0=npz['pd0']; d0=npz['d0']
    rbt.g=g; rbt.pd0=pd0; rbt.d0=d0

  npf = os.path.join(di,ddir,fi+'_ukf.npz')
  if os.path.exists(npf):
    if dbg:
      print 'loading '+npf
    npz = np.load( npf ) 
    X = npz['X']; hz = npz['hz']
    rbt.X = X; rbt.hz = hz;
    s = util.Struct()
    s.read( os.path.join(di,ddir,fi+'_ukf.py'), locals={'array':np.array})
    rbt.j = s.j; rbt.u = s.u; 

