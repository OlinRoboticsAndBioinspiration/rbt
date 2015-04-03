import os
import numpy as np
from util import files,geom,num,util
import matplotlib.pyplot as plt
from scipy import optimize

def line_angle(self, window=40, dbg=False, **kwds):
  """
  Estimate curvature of the data by fitting 2 lines and getting angle

  Prerequisites: {dat ukf | load}
    Optionally crop

  Effects:
    Sets the following metrics
    TODO fill in metrics
  """


  if self.is_valid == False:
      print "Got invalid, skipping"
      return
  t = self.t; X = self.X; j = self.j; u = self.u
  hz = self.hz
  if self.start_trial:
    #X = self.X[self.start_trial : self.stop_trial, ...]
    t = t[self.start_trial : self.stop_trial]
  else:
    print "WARNING, not cropping for line angle. Run crop first"
  N,_ = X.shape;
  #use the cropped version

  def rolling(series, func, win):
      num_curva = np.shape(series)[0]
      output = np.zeros((num_curva - win*2, 1))
      for i in range(win, num_curva -win):
        segment = series[i-win:i+win, :]
        output[i-win] = func(segment)
      return output

  def line_angle(segment):
      win = 3
      x1 = np.mean(segment[0:win, :], axis=0)
      x2 = np.mean(segment[len(segment)/2-win:len(segment)/2+win, :], axis=0)
      x3 = np.mean(segment[len(segment)-win-1:len(segment)-1, :], axis=0)
      a = (x1[0] - x2[0], x1[1] - x2[1], 0)
      b = (x3[0] - x2[0], x3[1] - x2[1], 0)
      adist= np.sqrt(a[0] * a[0] + a[1] * a[1])
      bdist= np.sqrt(b[0] * b[0] + b[1] * b[1])
      if adist== 0 or bdist== 0:
        return 0
      a = a / adist
      b = b / bdist
      res = np.cross(a, b)[2]
      return res

  imin = self.start_trial-window
  imax = self.stop_trial+window

  x_pos = X[imin:imax, self.j['x']]
  y_pos = X[imin:imax, self.j['y']]

  stack = np.vstack((x_pos, y_pos)).T
  line_angles = rolling(stack, line_angle, window)
  scaled_t = t[window:-window]

  mean_angles = np.mean(line_angles)
  median_angles = np.median(line_angles)
  std_angles = np.std(line_angles)

  samps = 5
  samps = 20
  size = line_angles.shape[0]/samps
  samples_means = [np.mean(line_angles[(x)*size: (x+1)*size]) for x in range(samps)]
  samples_stds= [np.std(line_angles[(x)*size: (x+1)*size]) for x in range(samps)]
  self.metrics_data["line_angle_cum_mean"] = mean_angles
  self.metrics_data["line_angle_cum_median"] = median_angles
  for i in range(samps):
    self.metrics_data["line_angle_mean_" + str(i)] = samples_means[i]
    self.metrics_data["line_angle_std_" + str(i)] = samples_means[i]
  self.metrics_data["line_angle_cum_std"] = std_angles

  self.line_angle = line_angles
  self.line_angle_t = scaled_t

  if dbg == True:
      plt.figure()
      plt.plot(scaled_t, line_angles)
      plt.title("Line angles vs time")
      plt.figure()
      plt.hist(line_angles, bins=30)
      plt.title("Line angle hist")
      plt.show()
