import os
import numpy as np
from util import files,geom,num,util
from scipy.signal import gaussian
import matplotlib
from matplotlib import pyplot as plt

def crop(self, std=40, window=300, find_at_speed=160, dbg=False,**kwds):
  """
  Estimates when the robot starts and stops running. Tries to remove start up noise as well as ending irregularities.

  Prerequisites: {ukf| load}
  Writes:
        self.start_trial
        self.stop_trial
        (self.is_valid)
  """
  t = self.t; X = self.X; N,_ = X.shape; j = self.j; u = self.u
  working_hz = np.mean((1./np.diff(t)))
  speed = np.sqrt(np.diff(X[...,j['x']])**2 + np.diff(X[...,j['y']])**2)*working_hz
  speed = np.clip(speed, 0, 1500)
  #window = 80
  gaus = gaussian(window, std)
  #normalize
  gaus /= np.mean(gaus)
  gaus /= window

  speed_conv = np.convolve(speed, gaus, mode="same")
  #sketch root finding... and cropping....
  #margin_seconds = (.4, .2)
  margin_seconds = (.6, .2)

  def bad_root_find():
    split_seconds = 2
    first_bit = np.abs(speed_conv[0:working_hz*split_seconds] - find_at_speed)
    first_idx = np.argmin(first_bit)
    second_bit = np.abs(speed_conv[working_hz*split_seconds:] - find_at_speed)
    second_idx = np.argmin(second_bit) + split_seconds * working_hz
    return first_idx, second_idx

  def root_find():
    split_seconds = 2
    first_idx = 0
    second_idx = 0
    for idx in range(int(working_hz*split_seconds)):
      first_idx = idx
      if speed_conv[idx] >= find_at_speed:
        break
    for idx in range(int(working_hz*split_seconds), len(speed_conv)):
      second_idx = idx
      if speed_conv[idx] <= find_at_speed:
        break
    return first_idx, second_idx

  first_idx, second_idx = root_find()


  first_idx += margin_seconds[0] * working_hz
  second_idx -= margin_seconds[1] * working_hz

  if (second_idx - first_idx <= 250):
    print "ERROR on file" + self.fi
    print "Crop didn't go according to plan"
    print "Consider removing or looking at"
    first_idx = 0
    second_idx = N
    self.is_valid = False

  if dbg:
    max_speed = np.max(speed_conv)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(t[1:N], speed)
    ax.plot([t[first_idx], t[first_idx]], [0, max_speed], linewidth=3.0)
    ax.plot([t[second_idx], t[second_idx]], [0, max_speed], linewidth=3.0)
    ax.plot([t[1], t[N-1]], [find_at_speed, find_at_speed], linewidth=2.0)
    ax.set_ylim(-100.,2100.)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Speed (mm/s)")
    ax.set_title("Cropping bars on time unsmoothed")

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(t[0:N-1], speed_conv)
    ax.plot([t[first_idx], t[first_idx]], [0, max_speed], linewidth=3.0)
    ax.plot([t[second_idx], t[second_idx]], [0, max_speed], linewidth=3.0)
    ax.plot([t[1], t[N-1]], [find_at_speed, find_at_speed], linewidth=2.0)
    ax.set_ylim(-100.,2100.)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Speed (mm/s)")
    ax.set_title("Cropping bars on time smoothed")

    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    x = X[...,j['x']]
    y = X[...,j['y']]
    ax.plot(x,y)
    ax.plot([x[first_idx]], [y[first_idx]], 'ro')
    ax.plot([x[second_idx]], [y[second_idx]], 'bo')

    plt.show()




  self.start_trial = int(first_idx)
  self.stop_trial = int(second_idx)
