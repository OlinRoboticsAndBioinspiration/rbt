import os
import numpy as np
from util import files,geom,num,util
from time import time

deg2rad = np.pi / 180.
rad2deg = 180. / np.pi
ddir = 'dat'

def ukf(self,ord=2,N=np.inf,Ninit=20,viz=0,**kwds):
  """
  Use UKF to track previously-loaded trajectory

  Inputs:
    (optional)
    ord - int - order of state derivative to track
    N - int - max number of samples to track
    Ninit - int - # of init iterations for ukf
    viz - int - # of samps to skip between vizualization; 0 to disable
    plts - [str,...] - list of plots to generate

  Outputs:
    X - N x 6 - rigid body state estimate at each sample

  Effects:
    - assigns self.X
    - saves X to fi+'_ukf.py' and fi+'_ukf.npz'
  """
  # unpack data
  t = self.t; d = self.d; g = self.g; fi = self.fi; hz=self.hz
  nn = np.logical_not( np.any( np.isnan(d[:,:,0]), axis=1) ).nonzero()[0]
  assert nn.size > 0
  n = 0
  if nn[0] > 0:
    n = nn[0]
    print self.trk+' not visible until sample #%d; trimming data' % n
    t = t[n:]; d = d[n:,:,:]
  di,fi = os.path.split(fi)
  N0,_,_ = d.shape; N = min(N,N0)

  # init ukf
  import uk
  from uk import body
  from uk import pts
  X0 = np.hstack( ( np.zeros(2), 2*np.random.rand(), # pitch,roll,yaw
                   num.nanmean(d[:100,:,:],axis=0).mean(axis=0) ) ) # xyz
  Qd = ( np.hstack( (np.array([1,1,1])*2e-3, np.array([1,1,1])*5e+0) ) )
  for o in range(ord-1):
    X0 = np.hstack( ( X0, np.zeros(6) ) )
    Qd = np.hstack( ( Qd, Qd[-6:]*1e-1) )
  b = body.Mocap( X0, g.T, viz=viz, Qd=Qd ); 
  b.Ninit = Ninit;
  print 'running ukf on %d samps' % N; ti = time()
  t = t[:N]
  j = dict(pitch=0,roll=1,yaw=2,x=3,y=4,z=5,dpitch=6, droll=7, dyaw=8, dx=9, dy=10, dz=11)
  X = uk.mocap( b, np.swapaxes(d[:N,:,:],1,2) ).T
  N,M = X.shape
  X = np.vstack(( np.nan*np.zeros((n,M)), X ))
  X[:,0:3] *= rad2deg
  u = dict(pitch='deg',roll='deg',yaw='deg',x='mm',y='mm',z='mm')
  self.j = j; self.u = u; self.X = X
  print '%0.1f sec' % (time() - ti)
  s = util.Struct(X0=X0,Qd=Qd,ord=ord,b=b,j=j,u=u)
  dir = os.path.join(di,ddir)
  if not os.path.exists( dir ):
    os.mkdir( dir )
  s.write( os.path.join(dir,fi+'_ukf.py') )
  np.savez(os.path.join(dir,fi+'_ukf.npz'),X=X,hz=hz)

  return X
