import os
import numpy as np
from util import files,geom,num,util
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

fmts = ['png']
pdir = 'plt'

def plt(self,fmts=fmts,plts=['3d','2d','pd','xyz0','xyz','dxyz','pry','dpry', 'exp'],
        save=True, show=True, crop=False, **kwds):
  import matplotlib.pyplot as plt
  """
  Plot trajectory data 
  Inputs:
    (optional)
    fmts - [str,...] - list of formats to export figures
    plts - [str,...] - list of plots to generate

  Effects:
    - generates & saves plots
  """
  di,fi = os.path.split(self.fi)
  dir = os.path.join(di,pdir)
  if save:
    if not os.path.exists( dir ):
      os.mkdir( dir )
  F = 1 # figure counter
  # processed marker data
  if self.d is not None:
    t = self.t; d = self.d
    # 3d
    if '3d' in plts:
      fig = plt.figure(F); fig.clf(); F += 1
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(d[::10,:,0],
                 d[::10,:,1],
                 d[::10,:,2])
      ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
      ax.set_title("Marker Trajectory")
      #ax.set_title('floor normal = %s'%np.array_str(n,precision=2))
      #ax.view_init(elev=0.,azim=-115.)
      ax.view_init(elev=90.,azim=90.)
      if save:
        for fmt in fmts:
          fig.savefig(os.path.join(di,pdir,fi+'_dat-3d.'+fmt))

    # xyz0
    if 'xyz0' in plts:
      fig = plt.figure(F); fig.clf(); F += 1
      ax = fig.add_subplot(311); ax.grid('on')
      ax.set_title('$x$, $y$, $z$ plot')
      ax.plot(t,d[...,0])
      ax.set_ylabel('$x$ (mm)')
      ax = fig.add_subplot(312); ax.grid('on')
      ax.plot(t,d[...,1])
      ax.set_ylabel('$y$ (mm)')
      ax = fig.add_subplot(313); ax.grid('on')
      ax.plot(t,d[...,2])
      ax.set_ylabel('$z$ (mm)')
      ax.set_xlabel('time (sec)')
      if save:
        for fmt in fmts:
          fig.savefig(os.path.join(di,pdir,fi+'_dat-xyz0.'+fmt))
  if hasattr(self,'pd0') and self.pd0 is not None and self.d0 is not None:
    pd0 = self.pd0; d0 = self.d0
    # pd
    if 'pd' in plts:
      fig = plt.figure(F); fig.clf(); F += 1
      ax = fig.add_subplot(111); ax.grid('on')
      ax.plot(t,pd0 - d0);
      ax.set_ylim(-5,5)
      ax.set_xlabel('time (sec)'); ax.set_ylabel('distance (mm)')
      ax.set_title('pairwise distances')
      if save:
        for fmt in fmts:
          fig.savefig(os.path.join(di,pdir,fi+'_dat-pd.'+fmt))
    if 'pdhist' in plts and self.X is not None:
      t = self.t; X = self.X; N,_ = X.shape; j = self.j; u = self.u
      x = X[...,j['x']]; y = X[...,j['y']];
      e = np.abs( pd0[:x.size,0] - d0[0] )
      nn = ((1 - np.isnan(e)) * (1 - np.isnan(x))).nonzero()
      x = x[nn]; y = y[nn]; e = e[nn]
      N = 10; de = 5; dd = 1000
      bins = [ 
               #np.linspace(x.min(),x.max(),num=N),
               #np.linspace(y.min(),y.max(),num=N),
               np.linspace(-dd,dd,num=N),
               np.linspace(-dd,dd,num=N),
               np.linspace(0.,de,num=10*de) 
             ]
      samps = np.c_[x,y,e]
      H,_ = np.histogramdd( samps, bins )
      w = bins[2][1:]# + np.diff(bins[2]))
      im = np.sum( H * w, axis=2 )

      fig = plt.figure(F); fig.clf(); F += 1
      ax = fig.add_subplot(111)
      plt.imshow( im, interpolation='nearest' )
      ax.set_xticks( range(N-1)[::2] )
      ax.set_xticklabels(['%0.0f' % xe for xe in np.linspace(-dd,dd,num=N/2)])
      ax.set_yticks( range(N-1)[::2] )
      ax.set_yticklabels(['%0.0f' % xe for xe in np.linspace(-dd,dd,num=N/2)])
      ax.set_xlabel('$x$ (mm)'); ax.set_xlabel('$y$ (mm)')
      ax.set_title("pdhist")
      #1/0
  # ukf data
  if self.X is not None:
    t = self.t; X = self.X; N,_ = X.shape; j = self.j; u = self.u

    if crop:
        t = self.t[self.start_trial : self.stop_trial, ...]
        X = self.X[self.start_trial : self.stop_trial, ...]
    #s = util.Struct()
    #s.read(os.path.join(di,ddir,fi+'_ukf.py'),locals={'array':np.array})
    #j = s.j; u = s.u
    if self.d is not None and 'exp' in plts and hasattr(self,'pd0') and self.pd0 is not None and self.d0 is not None:
      pd0 = self.pd0; d0 = self.d0
      t = self.t; d = self.d
      hz = self.hz
      fig = plt.figure(F); fig.clf(); F += 1
      ax = fig.add_subplot(211); ax.grid('on')
      ax.set_title('%dhz' % hz)
      spd = np.sqrt(np.diff(X[...,j['x']])**2 + np.diff(X[...,j['y']])**2)*hz
      ax.plot(t[1:N],spd,'b')
      ax.set_ylabel('speed (%s / sec)'%u['x'])
      ax.set_ylim(-100.,2100.)
      ax = fig.add_subplot(212); ax.grid('on')
      ax.plot(t,pd0 - d0);
      ax.set_ylim(-5,5)
      ax.set_ylabel('distance (mm)')
      ax.set_xlabel('time (sec)')
      if save:
        for fmt in fmts:
          fig.savefig(os.path.join(di,pdir,fi+'_ukf-exp.'+fmt))
    # xyz
    if 'xyz' in plts:
      fig = plt.figure(F); fig.clf(); F += 1
      ax = fig.add_subplot(311); ax.grid('on')
      ax.set_title('$x$, $y$, $z$ plot')
      ax.plot(t[:N],X[...,j['x']],'b')
      ax.set_ylabel('$x$ (%s)'%u['x'])
      ax = fig.add_subplot(312); ax.grid('on')
      ax.plot(t[:N],X[...,j['y']],'g')
      ax.set_ylabel('$y$ (%s)'%u['y'])
      ax = fig.add_subplot(313); ax.grid('on')
      ax.plot(t[:N],X[...,j['z']],'r')
      ax.set_ylabel('$z$ (%s)'%u['z'])
      ax.set_xlabel('time (sec)')
      if save:
        for fmt in fmts:
          fig.savefig(os.path.join(di,pdir,fi+'_ukf-xyz.'+fmt))
    if '2d' in plts:

      fig = plt.figure(F); fig.clf(); F += 1
      ax = fig.add_subplot(111)
      ax.plot(X[...,j['x']],
                 X[...,j['y']])
      num_points = X[..., j["x"]].shape[0] / self.hz + 1
      print num_points
      print self.hz
      x_every = [X[int(indx * self.hz), j['x']] for indx in range(num_points)]
      y_every = [X[int(indx * self.hz), j['y']] for indx in range(num_points)]
      print x_every
      ax.plot(x_every, y_every, 'go')
      ax.plot(X[0, j['x']], X[0, j['y']], "ro")

      ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)');
      ax.set_title("Top Down Trajectory")
      if save:
        for fmt in fmts:
          fig.savefig(os.path.join(di,pdir,fi+'_dat-2d.'+fmt))
    # pry
    if 'pry' in plts:
      fig = plt.figure(F); fig.clf(); F += 1
      ax = fig.add_subplot(311); ax.grid('on')
      ax.set_title('pitch, roll, yaw plot')
      ax.plot(t[:N],X[...,j['pitch']],'b')
      ax.set_ylabel('pitch (%s)'%u['pitch'])
      ax = fig.add_subplot(312); ax.grid('on')
      ax.plot(t[:N],X[...,j['roll']],'g')
      ax.set_ylabel('roll (%s)'%u['roll'])
      ax = fig.add_subplot(313); ax.grid('on')
      ax.plot(t[:N],X[...,j['yaw']],'r')
      ax.set_ylabel('yaw (%s)'%u['yaw'])
      ax.set_xlabel('time (sec)')
      if save:
        for fmt in fmts:
          fig.savefig(os.path.join(di,pdir,fi+'_ukf-rpy.'+fmt))
        # pry
      if 'dpry' in plts:
        fig = plt.figure(F); fig.clf(); F += 1
        ax = fig.add_subplot(311); ax.grid('on')
        ax.set_title('$\dot{pitch}, $\dot{roll}, $\dot{yaw} plot')
        ax.plot(t[:N],X[...,j['dpitch']],'b')
        ax.set_ylabel('$\dot{pitch} (%s)'%u['pitch'])
        ax = fig.add_subplot(312); ax.grid('on')
        ax.plot(t[:N],X[...,j['droll']],'g')
        ax.set_ylabel('$\dot{roll} (%s)'%u['roll'])
        ax = fig.add_subplot(313); ax.grid('on')
        ax.plot(t[:N],X[...,j['dyaw']],'r')
        ax.set_ylabel('$\dot{yaw} (%s)'%u['yaw'])
        ax.set_xlabel('time (sec)')
        if save:
          for fmt in fmts:
            fig.savefig(os.path.join(di,pdir,fi+'_ukf-rpy.'+fmt))
    if X.shape[1] >= 12:
      # xyz
      if 'dxyz' in plts:
        fig = plt.figure(F); fig.clf(); F += 1
        ax = fig.add_subplot(311); ax.grid('on')
        ax.set_title('$\dot{x}$, $\dot{y}$, $\dot{z}$ plot')
        ax.plot(t[:N],X[...,j['x']+6],'b')
        ax.set_ylabel('$\dot{x}$ (%s/sample)'%u['x'])
        ax = fig.add_subplot(312); ax.grid('on')
        ax.plot(t[:N],X[...,j['y']+6],'g')
        ax.set_ylabel('$\dot{y}$ (%s/sample)'%u['y'])
        ax = fig.add_subplot(313); ax.grid('on')
        ax.plot(t[:N],X[...,j['z']+6],'r')
        ax.set_ylabel('$\dot{z}$ (%s/sample)'%u['z'])
        ax.set_xlabel('time (sec)')
        if save:
          for fmt in fmts:
            fig.savefig(os.path.join(di,pdir,fi+'_ukf-dxyz.'+fmt))

  if show:
      plt.show()

