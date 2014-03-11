import os
import numpy as np
import util

ddir = 'dat'
def plane(self,fmts=['png'],out1=50.,out2=15.,**kwds):
  """
  Fit ground plane to data

  Inputs
    fmts - [str,...] - list of figure formats to export
    out1,out2 - float - magic numbers for outlier rejection

  Usage 
    >> d.shape # raw data, N samples of M markers with 3 coordinates (x,y,z)
    (N, M, 3)
    >> c = np.dot(d, R.T) - t # rectified data

  Effects
    - generates fi+'_cal.py' file containing dict of R,t,n
  """
  # unpack data
  t = self.t; d = self.d; g = self.g; hz=self.hz
  di,fi = os.path.split(self.fi)
  nn = np.logical_not( np.any( np.isnan(d[:,:,0]), axis=1) ).nonzero()[0]
  assert nn.size > 0
  N,M,_ = d.shape

  # swap axes in mocap hardware-dependent way
  if self.dev == 'opti':
    R0 = np.array([[0,0,1],[1,0,0],[0,1,0]])
  elif self.dev == 'vicon':
    R0 = np.identity(3)
  # R0 in SO(3)
  assert ( ( np.all(np.dot(R0,R0.T) == np.identity(3)) ) 
          and ( np.linalg.det(R0) == 1.0 ) ) 

  d = np.dot(d, R0.T)

  # collect non-nan data
  x = d[...,0]; y = d[...,1]; z = d[...,2]
  nn = np.logical_not(np.isnan(x.flatten())).nonzero()
  p = np.vstack((x.flatten()[nn],
                 y.flatten()[nn],
                 z.flatten()[nn])).T
  m = p.mean(axis=0)
  p -= m 
  # remove outliers
  p = p[np.abs(p[:,2]) < out1,:]
  # fit plane to data (n is normal vec)
  n = util.geom.plane(p)
  # rotate normal vertical
  R = util.geom.orient(n)
  p = np.dot(p,R.T)
  # save plane data
  s = util.util.Struct(R=np.dot(R,R0),t=np.dot(m,R.T),n=n)
  s.write( os.path.join(di,ddir,fi+'_cal.py') )
