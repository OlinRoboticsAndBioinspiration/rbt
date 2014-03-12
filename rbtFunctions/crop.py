import os
import numpy as np
from util import files,geom,num,util
from scipy.signal import gaussian

def crop(self,dbg=False,**kwds):
  """
  Estimates when the robot starts and stops running. Tries to remove start up noise as well as ending irregularities.

  Prerequisites: {ukf| load}
  Writes: 
        self.start_trial
        self.stop_trial
        (self.is_valid)
  """
  t = self.t; X = self.X; N,_ = X.shape; j = self.j; u = self.u
  speed = np.sqrt(np.diff(X[...,j['x']])**2 + np.diff(X[...,j['y']])**2)*self.hz
  window = 80
  std = 40
  gaus = gaussian(window, std)
  #normalize
  gaus /= np.mean(gaus) 
  gaus /= window

  speed_conv = np.convolve(speed, gaus, mode="same")
  #sketch root finding... and cropping....
  find_at_speed = 100
  split_seconds = 2
  margin_seconds = (.4, .4)

  def bad_root_find():
    first_bit = np.abs(speed_conv[0:self.hz*split_seconds] - find_at_speed)
    first_idx = np.argmin(first_bit)
    second_bit = np.abs(speed_conv[self.hz*split_seconds:] - find_at_speed)
    second_idx = np.argmin(second_bit) + split_seconds * self.hz
    return first_idx, second_idx
  
  def root_find():

    first_idx = 0
    second_idx = 0
    for idx in range(self.hz*split_seconds):
      if speed_conv[idx] >= find_at_speed:
        first_idx = idx
        break
    for idx in range(self.hz*split_seconds, len(speed_conv)):
      if speed_conv[idx] <= find_at_speed:
        second_idx = idx
        break
    return first_idx, second_idx

  first_idx, second_idx = root_find()

  first_idx += margin_seconds[0] * self.hz
  second_idx -= margin_seconds[1] * self.hz

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
    ax.plot([first_idx / float(self.hz), first_idx / float(self.hz)], [0, max_speed], linewidth=3.0)
    ax.plot([second_idx / float(self.hz), second_idx / float(self.hz)], [0, max_speed], linewidth=3.0)
    ax.set_ylim(-100.,2100.)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Speed (mm/s)")
    ax.set_title("Cropping bars on time unsmoothed")
    
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(t[1:N], speed_conv)
    ax.plot([first_idx / float(self.hz), first_idx / float(self.hz)], [0, max_speed], linewidth=3.0)
    ax.plot([second_idx / float(self.hz), second_idx / float(self.hz)], [0, max_speed], linewidth=3.0)
    ax.set_ylim(-100.,2100.)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Speed (mm/s)")
    ax.set_title("Cropping bars on time smoothed")
    plt.show()

  self.start_trial = first_idx
  self.stop_trial = second_idx
