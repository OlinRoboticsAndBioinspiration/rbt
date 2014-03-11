import os
import numpy as np
import plane
import util

ddir = 'dat'
def geom(self,**kwds):
  """
  Fit rigid body geometry

  Effects:
    - assigns self.g
    - saves g to fi+'_geom.npz'
  """
  # unpack data
  di,fi = os.path.split(self.fi)
  d = self.d
  N,M,D = d.shape
  # samples where all features appear
  nn = np.logical_not( np.any(np.isnan(d[:,:,0]),axis=1) ).nonzero()[0]
  #assert nn.size > 0
  # fit geometry to pairwise distance data
  pd0 = []; ij0 = []
  for i,j in zip(*[list(a) for a in np.triu(np.ones((M,M)),1).nonzero()]):
    ij0.append([i,j])
    pd0.append(np.sqrt(np.sum((d[:,i,:] - d[:,j,:])**2,axis=1)))
  pd0 = np.array(pd0).T; 
  d0 = util.num.nanmean(pd0,axis=0); ij0 = np.array(ij0)
  self.pd0 = pd0; self.d0 = d0
  g0 = d[nn[0],:,:]

  # TODO: fix geometry fitting
  if 1:
    g = g0.copy()
  else:
    print 'fitting geom'; ti = time()
    g,info,flag = geom.fit( g0, ij0, d0 )
    print '%0.1f sec' % (time() - ti)
    pd = []; pd0 = []
    for i,j in zip(*[list(a) for a in np.triu(np.ones((M,M)),1).nonzero()]):
      pd.append( np.sqrt( np.sum((g[i,:] - g[j,:])**2) ) )
      pd0.append( np.sqrt( np.sum((g0[i,:] - g0[j,:])**2) ) )
    pd = np.array(pd).T; 
    pd0 = np.array(pd0).T; 
    
  # center and rotate geom flat 
  m = np.mean(g,axis=0)
  g = g - m
  n = util.geom.plane( g)
  R = util.geom.orient(n)
  g = np.dot(g,R.T)
  self.g = g
  # save data
  dir = os.path.join(di,ddir)
  if not os.path.exists( dir ):
    os.mkdir( dir )
  np.savez(os.path.join(dir,fi+'_geom.npz'),g=g,pd0=pd0,d0=d0)

