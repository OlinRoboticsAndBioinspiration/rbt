import os
import numpy as np
from util import files,geom,num,util
import matplotlib.pyplot as plt
from scipy import optimize

def circle_fit(self, dbg=False, **kwds):
  """
  Estimate curvature of the data by fitting many circles to different
  areas of the data.

  Prerequisites: {dat ukf | load}
    Optionally crop

  Effects:
    Sets the following metrics
      self.metrics_data["mean_circle_fit_radius"]
      self.metrics_data["mean_circle_fit_curvature"]
      self.metrics_data["median_circle_fit_curvature"]

  """
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
  window = 40
  circles = circle_fit_dist(x_pos, y_pos, window, stride)
  print "Circles shape", circles.shape
  print "X shape", X.shape
  if dbg == True:
      plt.figure()
      plt.plot(x_pos, y_pos, linewidth=3)
      xlim = plt.xlim()
      ylim = plt.ylim()

      period = np.linspace(0, np.pi*2, 100)
      for xc, yc, rc, distance, direction in circles[0:-1:10,...]:
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

  scaled_t = t[0:circles[..., 2].shape[0]*stride:stride]
  if dbg == True:
      plt.figure()
      plt.semilogy(scaled_t, dists)
      plt.xlabel("time")
      plt.ylabel("circle accuracy")
      plt.title("r^2 of circle fits")

  #trimmed with r^2 values
  curva_trimmed = np.copy(curvature)
  curva_trimmed[dists > 5000] = 0
  if dbg == True:
      plt.figure()
      plt.plot(scaled_t, curva_trimmed, 'r')
      plt.plot(scaled_t, curvature)
      plt.xlabel("time")
      plt.ylabel("circle curvature")
      plt.title("r^2 trimmed curvature")
      plt.show()

  #Sampled circle fit data

  #rolling median
  print len(curvature), "length of curvature"
  def rolling(series, func, win):
      num_curva = len(series)
      output = np.zeros((num_curva - win*2, 1))
      for i in range(win, num_curva -win):
        segment = series[i-win:i+win]
        output[i-win] = func(segment)
      return output
  win = 30
  curva_smooth= rolling(curvature, np.median, win)
  scaled_t = t[0:circles[..., 2].shape[0]*stride:stride]
  output_t = scaled_t[win:len(curvature)-win]

  if dbg==True:
      plt.figure()
      plt.plot(output_t, curva_smooth)
      ylim = plt.ylim()
      plt.plot(scaled_t, curvature)
      plt.ylim(ylim)
      plt.legend(["filtered", "raw"])
      plt.xlabel("time")
      plt.title("rolling median curvature")
  #calculate stability

  self.circle_fit_t = output_t
  self.circle_fit = curvature.T

  print "std of entire data", np.std(curvature)
  variance = rolling(curvature, np.std, win)
  mean_win = 10
  variance_smooth = rolling(variance, np.mean, mean_win)
  if dbg == True:
      plt.figure()
      plt.plot(output_t, variance)
      plt.plot(output_t[mean_win:-mean_win], variance_smooth)
      plt.title("rolling std")
      plt.ylabel("std")


  #partition into 4
  samples = len(variance_smooth)
  split = samples/4
  for k in range(4):
    #select lowest variance
    segment = variance_smooth[split*k:split*(k+1)]
    index_in_smoothed = np.argmin(segment) + split*k
    index_in_curva_smooth = index_in_smoothed + mean_win
    #get curvature at that point
    curva = curva_smooth[index_in_curva_smooth]
    #self.metrics_data["segment"+str(k)+"_circle_fit_curva"] = float(curva)
    self.metrics_data["segment"+str(k)+"_circle_fit_curva"] = float(np.mean(segment))
    self.metrics_data["segment"+str(k)+"_circle_fit_std"] = float(np.min(segment))



  if dbg == True:
      scaled_t = t[0:circles[..., 2].shape[0]*stride:stride]
      print mean_radi
      mean_radi = np.mean(radi)

      scaled_t = scaled_t[~is_outlier(curvature)]
      direction = direction[~is_outlier(curvature)]
      radi = radi[~is_outlier(curvature)]
      curvature = curvature[~is_outlier(curvature)]

      negradi = radi[direction == -1]
      negt = scaled_t[direction == -1]
      posradi = radi[direction == 1]
      post = scaled_t[direction == 1]

      plt.figure()
      plt.plot(scaled_t, curvature, 'o')
      plt.title("curvature vs time")
      plt.xlabel('time (s)')
      plt.ylabel("curvature (1/mm)")

      plt.figure()
      plt.hist(curvature, bins=100)
      print "Mean curvature", mean_curvature
      print "Median curvature", median_curvature

      #plt.hist(np.log(radi), bins=50)
      plt.show()


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

