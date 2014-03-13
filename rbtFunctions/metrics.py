import os
import numpy as np
from util import files,geom,num,util
from scipy.signal import gaussian

metrics_dir = 'metrics'
def metrics(self, dbg = False, **kwds):
  """
  Saves various statistics and metrics about the run. Writes metrics to file.

  Prerequisites: {ukf, load} crop mcu
  Effects: Writes metrics python dictionary
  """

  t = self.t; X = self.X; j = self.j; u = self.u
  hz = self.hz
  if self.start_trial:
    X = self.X[self.start_trial : self.stop_trial, ...]
  N,_ = X.shape;
  
  """if dbg:
    fig = plt.figure(1);
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
  """

  yaw = X[...,j["yaw"]]
  window = 140
  std = 400 
  gaus = gaussian(window, std)
  gaus /= np.mean(gaus)
  gaus /= float(window)
  dir_window = [1, 0, -1]
  smooth_yaw = np.convolve(yaw, gaus, mode="same")
  d_smooth_yaw = np.convolve(smooth_yaw, dir_window, mode="same") / 2.0 * float(hz)
  #lop of the ends to account for convolution irregularities
  d_smooth_yaw = d_smooth_yaw[window:-window]
  smooth_yaw = smooth_yaw[window:-window]
  N = N-window*2
  #get reasonable edges

  bin_low = np.percentile(d_smooth_yaw, 10)
  bin_high = np.percentile(d_smooth_yaw, 90)
  trimmed_d_smooth_yaw = d_smooth_yaw[d_smooth_yaw > bin_low]
  trimmed_d_smooth_yaw = d_smooth_yaw[trimmed_d_smooth_yaw < bin_high]

  bins = np.linspace(bin_low, bin_high, 20)
  data, bins = np.histogram(trimmed_d_smooth_yaw, bins)
  max_bin = np.argmax(data)
  hist_mode_dyaw = bins[max_bin]

  #Write some basic self.metrics_data
  self.metrics_data["mean_dyaw"] = np.mean(d_smooth_yaw)
  self.metrics_data["trimmed_mean_dyaw"] = np.mean(trimmed_d_smooth_yaw)
  self.metrics_data["hist_mode_dyaw"] = hist_mode_dyaw
  self.metrics_data["median_dyaw"] = np.median(d_smooth_yaw)
  self.metrics_data["mean_vb"] = np.mean(self.mcu_data[..., self.mcu_j["vb"]])
  self.metrics_data["median_vb"] = np.median(self.mcu_data[..., self.mcu_j["vb"]])
  self.metrics_data["hz"] = float(self.hz)
  self.metrics_data["mocap_points"] = N
  self.metrics_data["mcu_points"] = self.mcu_data.shape[0]
  self.metrics_data["is_valid"] = self.is_valid

  
  #clean up nans in metrics with zeros
  for key in self.metrics_data.keys():
      if self.metrics_data[key] == np.nan:
          self.metrics_data[key] = 0
          print "NAN found in key ", key

  #save them to file
  di,fi = os.path.split(self.fi)
  di = os.path.join(di, metrics_dir)
  if not os.path.exists(di):
    os.mkdir(di)

  self.metrics_data_file = open(os.path.join(di,fi+"_metrics.py"), "wb+")
  self.metrics_data_file.write("%s" % self.metrics_data)
  self.metrics_data_file.close()

  if dbg:
    plt.figure(2)
    plt.plot(t[:N], d_smooth_yaw)
    plt.xlabel("time (s)")
    plt.ylabel("degrees/s")
    plt.title("Derivative Gaussianed Yaw")
    bin_low = np.percentile(d_smooth_yaw, 10)
    bin_high = np.percentile(d_smooth_yaw, 90)

    plt.figure(3)
    plt.hist(d_smooth_yaw, np.linspace(bin_low, bin_high, 100))
    plt.xlabel("degree / second")
    plt.ylabel("freqency")
    plt.title("Histogram derivative Yaw")
    plt.figure(4)
    plt.plot(t[:N], smooth_yaw)
    plt.plot(t[:len(yaw)], yaw)
    plt.legend(["Smoothed yaw", "yaw"])
    plt.xlabel("time (s)")
    plt.ylabel("angle (degrees)")
    plt.title("Gaussianed Yaw")
    plt.figure(5)
    x = X[...,j['x']]
    y = X[...,j['y']]
    plt.plot(x, y)
    time = len(x)
    num_seconds = time / float(self.hz)
    interval = .5
    points = num_seconds / interval
    bits_idx = [point * interval * float(self.hz) for point in range(int(points))]
    bits_x = x[bits_idx]
    bits_y = y[bits_idx]
    plt.plot(bits_x, bits_y, 'ro')
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("2d Trajectory")
    
    plt.figure(6)
    vbat = self.mcu_data[..., self.mcu_j["vb"]]
    min_vbat = np.percentile(vbat, 5)
    max_vbat = np.percentile(vbat, 95)
    plt.hist(vbat, np.linspace(min_vbat, max_vbat, 50))
    plt.xlabel("voltage (V)")
    plt.ylabel("freqency")
    plt.title("Histogram battery voltage")

    plt.show()
