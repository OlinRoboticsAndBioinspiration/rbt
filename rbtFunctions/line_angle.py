import os
import numpy as np
from util import files,geom,num,util
import matplotlib.pyplot as plt
from scipy import optimize

def line_angle(self, dbg=False, **kwds):
  """
  Estimate curvature of the data by fitting 2 lines and getting angle

  Prerequisites: {dat ukf | load}
    Optionally crop

  Effects:
    Sets the following metrics
    TODO fill in metrics
  """


  t = self.t; X = self.X; j = self.j; u = self.u
  hz = self.hz
  if self.start_trial:
    X = self.X[self.start_trial : self.stop_trial, ...]
    t = t[self.start_trial : self.stop_trial]
  else:
    print "WARNING, not cropping for circle fit. Run crop first"
  N,_ = X.shape;
  #use the cropped version
  x_pos = X[..., self.j['x']]
  y_pos = X[..., self.j['y']]

  window = 40

  def rolling(series, func, win):
      num_curva = np.shape(series)[0]
      output = np.zeros((num_curva - win*2, 1))
      for i in range(win, num_curva -win):
        segment = series[i-win:i+win, :]
        output[i-win] = func(segment)
      return output

  def line_angle(segment):
      x1 = segment[0, :]
      x2 = segment[len(segment)/2, :]
      x3 = segment[-1, :]
      a = (x1[0] - x2[0], x1[1] - x2[1], 0)
      b = (x3[0] - x2[0], x3[1] - x2[1], 0)
      res = np.cross(a, b)[2]
      return res
  stack = np.vstack((x_pos, y_pos)).T
  line_angles = rolling(stack, line_angle, window)
  scaled_t = t[window:-window]

  mean_angles = np.mean(line_angles)
  std_angles = np.std(line_angles)

  samps = 5
  samps = 20
  size = line_angles.shape[0]/samps
  samples_means = [np.mean(line_angles[(x)*size: (x+1)*size]) for x in range(samps)]
  print samples_means
  self.metrics_data["line_angle_mean"] = mean_angles
  for i in range(samps):
    self.metrics_data["line_angle_mean_" + str(i)] = samples_means[i]
  self.metrics_data["line_angle_stds"] = std_angles

  if dbg == True:
      plt.figure()
      plt.plot(scaled_t, line_angles)
      plt.figure()
      plt.hist(line_angles, bins=30)
      plt.show()
