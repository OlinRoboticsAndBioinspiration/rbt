import os
import numpy as np
from util import files,geom,num,util
from time import time


ddir = 'dat'
dsfx = {'opti':'_mocap.csv','vicon':'.dcr','phase':'.txt'}

m2mm = 1000.
sync = ['dat']

def dat(self,N=np.inf,hz0=None,save=True,dbg=True,**kwds):

  import rbt
  """
  Load raw mocap rigid body data

  Inputs
    (optional)
    N - int - max number of data samples to read
    hz - int - sample rate

  Effects:
    - assigns self.t,.d,.g
    - saves t,d to fi+'_dat.py' and fi+'_dat.npz'
  """
  # unpack data
  fi = self.fi; trk = self.trk
  # read data
  di,fi = os.path.split(self.fi)
  dev = self.dev; sfx = dsfx[dev]
  if dbg:
    print 'reading '+os.path.join(di,fi+sfx); ti = time()

  if dev == 'phase':
    _,s,a,r = fi.strip(sfx).split('_')
    self.trk = s+a+r
    if sfx == '.nik':
      d0 = np.loadtxt(os.path.join(di,fi+sfx))
      t = np.arange(d0.shape[0]) / 480. # fake time samples
      #t = d_[:,-1]; d0 = d_[:,:-2]; d0[d0 == 0.] = np.nan
      #if np.allclose(d_[:,-2],np.arange(d_.shape[0])):
      #  t = d_[:,-1]
      #else:
      #  t = d_[:,-2] + d_[:,-1]*1e-6; 
      #d0 = d_[:,:-2]; d0[d0 == 0.] = np.nan
      if dbg:
        print '%0.1f sec' % (time() - ti)
      if N < np.inf:
        t = t[:N]; d0 = d0[:N]
      N0,M0 = d0.shape; D = 3; M = M0 / D
      #d0.shape = (N0,M,D)
      d0.shape = (N0,D,M)
      #d0 = d0.transpose(0,2,1)
      d0 = d0 / 10. # convert from mm to cm
      #d[:,0] = -d[:,0] # flip x axis
      # insert nan's for missing samples
      dt = np.diff(t)#; dt = dt[(1-np.isnan(dt)).nonzero()]
      hz = int(np.round(1./np.median(dt)))
      if hz0:
        assert hz0 == hz
      else:
        if dbg:
          print 'assuming hz = %d' % hz
      N = int( np.ceil( (t[-1] - t[0]) * hz) ) + 1
      d = np.nan*np.zeros((N,M,D))
      j = np.array( np.round( (t - t[0]) * hz ), dtype=np.int )
      t = np.arange(N) / float(hz)
      # enforce uniform time increments
      d[j,:,:] = d0
      ## remove unobserved features
      #nn = np.logical_not( np.all(np.isnan(d[:,:,0]),axis=0) ).nonzero()[0]
      #if dbg:
      #  print 'keeping observed markers %s' % nn
      #d = d[:,nn,:]
    elif sfx == '.rob':
      d_ = np.loadtxt(os.path.join(di,fi+sfx))
      if np.allclose(d_[:,-2],np.arange(d_.shape[0])):
        t = d_[:,-1]
      else:
        t = d_[:,-2] + d_[:,-1]*1e-6; 
      d0 = d_[:,:-2]; d0[d0 == 0.] = np.nan
      if dbg:
        print '%0.1f sec' % (time() - ti)
      if N < np.inf:
        t = t[:N]; d0 = d0[:N]
      N0,M0 = d0.shape; D = 3; M = M0 / D
      d0.shape = (N0,M,D)
      #d0.shape = (N0,D,M); d0 = d0.transpose(0,2,1)
      d0 = d0 / 1. # convert from mm to cm
      dt = np.diff(t)#; dt = dt[(1-np.isnan(dt)).nonzero()]
      hz = int(np.round(1./np.median(dt)))
      if hz0:
        assert hz0 == hz
      else:
        if dbg:
          print 'measured hz = %d; setting hz = 480' % hz
      hz = 480
      N = int( np.ceil( (t[-1] - t[0]) * hz) ) + 1
      d = np.nan*np.zeros((N,M,D))
      j = np.array( np.round( (t - t[0]) * hz ), dtype=np.int )
      t = np.arange(N) / float(hz)
      # enforce uniform time increments
      #d[j,:,:] = d0
      d = d0
    elif sfx == '.txt':
      d_ = np.loadtxt(os.path.join(di,fi+sfx))
      if np.allclose(d_[:,-2],np.arange(d_.shape[0])):
        t = d_[:,-1]
      else:
        t = d_[:,-2] + d_[:,-1]*1e-6; 
      d0 = d_[:,:-2]; d0[d0 == 0.] = np.nan
      if dbg:
        print '%0.1f sec' % (time() - ti)
      if N < np.inf:
        t = t[:N]; d0 = d0[:N]
      N0,M0 = d0.shape; D = 3; M = M0 / D
      d0.shape = (N0,M,D)
      #d0.shape = (N0,D,M); d0 = d0.transpose(0,2,1)
      d0 = d0 / 10. # convert from mm to cm
      d0 = d0[...,[0,2,1]] # exchange y and z
      dt = np.diff(t)#; dt = dt[(1-np.isnan(dt)).nonzero()]
      hz = int(np.round(1./np.median(dt)))
      if hz0:
        assert hz0 == hz
      else:
        if dbg:
          print 'measured hz = %d; setting hz = 480' % hz
      hz = 480
      N = int( np.ceil( (t[-1] - t[0]) * hz) ) + 1
      d = np.nan*np.zeros((N,M,D))
      j = np.array( np.round( (t - t[0]) * hz ), dtype=np.int )

      t = np.arange(N) / float(hz)
      # enforce uniform time increments
      d[j,:,:] = d0
      ## remove unobserved features
      #nn = np.logical_not( np.all(np.isnan(d[:,:,0]),axis=0) ).nonzero()[0]
      #if dbg:
      #  print 'keeping observed markers %s' % nn
      #d = d[:,nn,:]
    else:
      import c3d
      with open(os.path.join(di,fi+sfx), 'rb') as h:
          r = c3d.Reader(h)
          d = np.dstack([p for p,_ in r.read_frames()])[:,:3,:].T
      
      t = np.arange(d.shape[0]) / 480. # fake time samples
      d[d == 0.0] = np.nan # missing observations
      d = d / 10. # convert from mm to cm
      #d[:,0] = -d[:,0] # flip x axis

  elif dev == 'opti':
    from mocap.python import optitrack as opti
    run = opti.Run()
    run.ReadFile(di,fi+sfx,N=N)
    if dbg:
      print '%0.1f sec' % (time() - ti)
    # extract data from trackable
    if trk:
      t,d0 = run.trk(trk)
    else:
      t,d0,_,_ = run.data()
    if N < np.inf:
      t = t[:N]; d0 = d0[:N]

    #Trim out nans
    nn = np.logical_not(np.isnan(t))
    t = t[nn]; d0 = d0[nn,...]

    d0 *= m2mm
    N0,M,D = d0.shape
    # insert nan's for missing samples
    dt = np.diff(t)#; dt = dt[(1-np.isnan(dt)).nonzero()]
    hz = int(np.round(1./np.median(dt)))

    #clean data

    if hz0:
      assert hz0 == hz
    else:
      if dbg:
        print 'assuming hz = %d' % hz
    N = int( np.ceil( (t[-1] - t[0]) * hz) ) + 1
    d = np.nan*np.zeros((N,M,D))
    j = np.array( np.round( (t - t[0]) * hz ), dtype=np.int )
    t = np.arange(N) / float(hz)
    # enforce uniform time increments
    d[j,...] = d0
    if not( trk == 'l' ) and not( trk == 'r' ):
      try:
        # align time samples with sync electronics
        args = dict(fi=self.fi,dev='opti',procs=sync,dbg=False,save=False)
        l = rbt.rbt.do_(trk='l',**args).d
        l = np.mean( np.isnan( np.reshape( l, (l.shape[0],-1) ) ), axis=1 )
        r = rbt.rbt.do_(trk='r',**args).d
        r = np.mean( np.isnan( np.reshape( r, (r.shape[0],-1) ) ), axis=1 )
        j = (.5*(l + r) < .9).nonzero()[0]
        t = t[j] - t[j[0]]; d = d[j,...]
      except AssertionError:
        pass # l or r trackable not found


  elif dev == 'vicon':
    from shrevz import viconparser as vp
    p = vp.ViconParser()
    p.load(os.path.join(di,fi))
    self.p = p
    if dbg:
      print '%0.1f sec' % (time() - ti)
    # extract data from trackable
    t0,d0 = p.t.flatten(),p.xyz
    hz = p.fps
    N0,M,D = d0.shape
    # insert nan's for missing samples
    N = int( np.ceil( t0[-1] - t0[0] ) ) + 1
    d = np.nan*np.zeros((N,M,D))
    j = np.array( np.round( t0 - t0[0] ), dtype=np.int )
    t = np.arange(N) / float(hz)
    d[j,:,:] = d0

  if dbg:
    print 'd0.shape = %s, d.shape = %s' % (d0.shape,d.shape)
  self.t = t; self.d = d;
  self.hz = hz; self.trk = trk;
  if save:
    # save data
    s = util.Struct(hz=hz,trk=trk,dev=dev)
    ndir = os.path.join(di,ddir)
    if not os.path.exists( ndir ):
      os.mkdir( ndir )
    s.write( os.path.join(ndir,fi+'_dat.py') )
    np.savez(os.path.join(ndir,fi+'_dat.npz'),t=t,d=d,hz=hz)

