import os
import numpy as np
from util import files,geom,num,util
import matplotlib.pyplot as plt
from scipy import optimize

# taken from http://wiki.scipy.org/Cookbook/Least_Squares_Circle
def alg_fit_circle(x, y):
  # coordinates of the barycenter
  x_m = np.mean(x)
  y_m = np.mean(y)
  # calculation of the reduced coordinates
  u = x - x_m
  v = y - y_m
  # linear system defining the center in reduced coordinates (uc, vc):
  #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
  #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
  Suv  = np.sum(u*v)
  Suu  = np.sum(u**2)
  Svv  = np.sum(v**2)
  Suuv = np.sum(u**2 * v)
  Suvv = np.sum(u * v**2)
  Suuu = np.sum(u**3)
  Svvv = np.sum(v**3)
  # Solving the linear system
  A = np.array([ [ Suu, Suv ], [Suv, Svv]])
  B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
  uc, vc = np.linalg.solve(A, B)
  xc = x_m + uc
  yc = y_m + vc
  # Calculation of all distances from the center (xc, yc)
  Ri      = np.sqrt((x-xc)**2 + (y-yc)**2)
  R       = np.mean(Ri)
  #residu  = sum((Ri-R)**2)
  #residu2 = sum((Ri**2-R**2)**2)
  return (xc, yc, R)

def leastsq_fit_circle(x, y):
  x_m = np.mean(x)
  y_m = np.mean(y)
  def calc_R(xc, yc):
      """ calculate the distance of each 2D points from the center (xc, yc) """
      return np.sqrt((x-xc)**2 + (y-yc)**2)

  def f_2(c):
      """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
      Ri = calc_R(*c)
      return Ri - Ri.mean()

  center_estimate = x_m, y_m
  center, ier = optimize.leastsq(f_2, center_estimate)

  xc, yc = center
  Ri       = calc_R(*center)
  R        = Ri.mean()
  residu   = np.sum((Ri - R)**2)
  return (xc, yc, R)

def circle_fit(self, dbg=False, **kwds):
  def circle_fit_dist(x, y, length, stride):
    valid_starts = (x.shape[0] - length)
    def get_radi(start):
      end = start+length
      trial_x = x[start:end]
      trial_y = y[start:end]
      (xc, yc, rc) = leastsq_fit_circle(trial_x, trial_y)
      distance = np.sqrt(np.mean((trial_x-xc) ** 2 + (trial_y - yc) ** 2))
      theta_start = np.arctan2(trial_x[0], trial_y[0])
      theta_end = np.arctan2(trial_x[-1], trial_y[-1])
      
      theta_start = np.arctan2((trial_x[0]-xc)/rc, (trial_y[0]-yc)/rc)
      theta_end = np.arctan2((trial_x[-1]-xc)/rc, (trial_y[-1]-yc)/rc)
      direction = np.sign(theta_start - theta_end)
      if direction == np.nan:
        direction = 0
      return (xc, yc, rc, distance, direction)

    circles = [get_radi(indx) for indx in range(0, valid_starts, stride)]

    return np.vstack(circles)

  t = self.t; X = self.X; j = self.j; u = self.u
  hz = self.hz
  if self.start_trial:
    X = self.X[self.start_trial : self.stop_trial, ...]
  else:
    print "WARNING, not cropping for circle fit. Run crop first"
  N,_ = X.shape;
  #use the cropped version
  x_pos = X[..., self.j['x']]
  y_pos = X[..., self.j['y']]
  (xc, yc, rc)= leastsq_fit_circle(x_pos, y_pos)


  stride = 1
  #window = 50
  window = 20
  #window = 30
  #window = 100
  circles = circle_fit_dist(x_pos, y_pos, window, stride)
  print "Circles shape", circles.shape
  print "X shape", X.shape
  if dbg == True:
      plt.figure()
      plt.plot(x_pos, y_pos, linewidth=3)
      xlim = plt.xlim()
      ylim = plt.ylim()

      period = np.linspace(0, np.pi*2, 100)
      for xc, yc, rc, distance, direction in circles[0:-1:3,...]:
          plt.plot(np.sin(period)*rc + xc, np.cos(period)*rc + yc)
      square_lim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))

      plt.xlim(square_lim)
      plt.ylim(square_lim)
      plt.plot(x_pos, y_pos, 'r', linewidth=5)
      plt.plot(x_pos, y_pos, 'k', linewidth=3)

  # from
  # http://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting
  def is_outlier(points, thresh=3.5):
      if len(points.shape) == 1:
          points = points[:,None]
      median = np.median(points, axis=0)
      diff = np.sum((points - median)**2, axis=-1)
      diff = np.sqrt(diff)
      med_abs_deviation = np.median(diff)
      modified_z_score = 0.6745 * diff / med_abs_deviation
      return modified_z_score > thresh
  radi = circles[..., 2]
  dists = circles[..., 3]
  direction = circles[..., 4]

  curvature = 1/(radi * direction)
  outlier_curvature = curvature[~is_outlier(curvature)]

  mean_radi = np.mean(radi * direction)
  self.metrics_data["mean_circle_fit_radius"] = mean_radi
  mean_curvature = np.mean(outlier_curvature)
  self.metrics_data["mean_circle_fit_curvature"] = mean_curvature
  median_curvature = np.median(outlier_curvature)
  self.metrics_data["median_circle_fit_curvature"] = median_curvature

  if dbg == True:
      plt.figure()
      scaled_t = t[0:circles[..., 2].shape[0]*stride:stride]
      plt.semilogy(scaled_t, radi)
      print mean_radi
      mean_radi = np.mean(radi)
      plt.semilogy([0, scaled_t[-1]+1], [mean_radi, mean_radi], "r", linewidth=3)
      plt.title("semilog radi")
      plt.xlabel("time (s)")
      plt.ylabel("radi (mm)")
      plt.figure()

      scaled_t = scaled_t[~is_outlier(curvature)]
      direction = direction[~is_outlier(curvature)]
      radi = radi[~is_outlier(curvature)]
      curvature = curvature[~is_outlier(curvature)]

      negradi = radi[direction == -1]
      negt = scaled_t[direction == -1]
      posradi = radi[direction == 1]
      post = scaled_t[direction == 1]
      plt.semilogy(negt, negradi, 'o', c=(1, 0, 0))
      plt.semilogy(post, posradi, 'o', c=(0, 0, 1))
      plt.legend(["neg direction", "pos direction"])
      #plt.semilogy([0, scaled_t[-1]+1], [mean_radi, mean_radi], "r", linewidth=3)
      plt.title("semilog radi shifted for neg")
      plt.xlabel("time (s)")
      plt.ylabel("radi (mm)")
      plt.figure()
      plt.plot(scaled_t, direction, 'o')
      plt.figure()
      plt.plot(scaled_t, curvature, 'o')
      plt.title("curvature vs time")
      plt.xlabel('time (s)')
      plt.ylabel("curvature (1/mm)")
      plt.figure()
      plt.hist(curvature, bins=100)
      print "Mean curvature", mean_curvature
      print "Median curvature", median_curvature
      #plt.figure()
      #plt.plot(scaled_t, 1.0/radi)
      #plt.figure()
      #plt.plot(scaled_t, radi*direction)
      #plt.errorbar(scaled_t, radi, yerr=dists)

      #plt.hist(np.log(radi), bins=50)
      plt.show()

