import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def standard_transform(self, dbg=False, **kwds):
  t = self.t; X = self.X; N,_ = X.shape; j = self.j; u = self.u
  x = X[..., j['x']]
  y = X[..., j['y']]

  x0 = x[0]
  y0 = y[0]
  x = x-x0
  y = y-y0

  xd = np.mean(x[75:125])
  yd = np.mean(y[75:125])
  radius = np.sqrt(xd**2 + yd**2)
  nxd = xd/radius
  nyd =  yd/radius
  theta = np.arctan2(nyd, nxd)

  rot = np.zeros((2,2))

  theta = theta - np.pi/2
  rot[0,0] = np.cos(theta)
  rot[1,1] = np.cos(theta)
  rot[0,1] = -np.sin(theta)
  rot[1,0] = np.sin(theta)

  trans = np.vstack((x,y)).T

  rotated = np.dot(trans,  rot)
  rx = rotated[..., 0]
  ry = rotated[..., 1]
  if dbg == True:
    print "bngle = ", theta/np.pi * 180
    plt.plot(x,y, 'ro')
    plt.plot(rx,ry,'go')
    plt.plot([0, nxd * 300], [0, nyd * 300],  'b', linewidth=4)
    plt.legend(["normal", "rotated", "direction"])
    plt.title("top down rotation test")
    plt.show()
  self.stand_x = rx
  self.stand_y = ry
