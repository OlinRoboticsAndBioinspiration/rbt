import os
import numpy as np
from util import files,geom,num,util

def mcu(self, dbg=False, **kwds):
  di,fi = os.path.split(self.fi)
  mcu_data = np.loadtxt(os.path.join(di,fi+"_mcu.csv"),delimiter=",")
  run_config = eval(open(os.path.join(di,fi+"_cfg.py")).read())
  mcu_j = run_config["rid"]["mcu_j"]

  #kill the nans by finding timestamps at 4294967295
  time = mcu_data[..., mcu_j['t']]
  dtime = np.diff(time)
  index = np.argmax(dtime)
  mcu_data = mcu_data[0:index, ...]

  # TODO start stop syncronization
  # Will need to scale aswell to match the two time scales possibly

  # spd = np.sqrt(np.diff(self.X[...,self.j['x']])**2 +
  #  np.diff(self.X[...,self.j['y']])**2)*self.hz
  # plt.plot(spd)
  # plt.ylim(0, 10)

  #TODO check this math. Taken from old benchmark script
  mcu_data[..., mcu_j['vb']] = mcu_data[..., mcu_j['vb']] * 2 * 3.3 / 1023.0
  if dbg:
    plt.plot(mcu_data[..., mcu_j['vb']])
    plt.xlabel("time (samples)")
    plt.ylabel("voltage")
    plt.title("Battery voltage over run")
    plt.show()
  self.mcu_data = mcu_data
  self.mcu_j = mcu_j
