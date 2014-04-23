import numpy as np
import os

def cfg_metrics(self, **kwds):
  """
  Writes metrics about the robot trial configuration.
  For now, just writing motor configuration thats set in run_config
  Prerequisites: 
  Effects: Sets motor_cfg_0 metric, and motor_cfg_1 metric
  """
  di,fi = os.path.split(self.fi)
  run_config = eval(open(os.path.join(di,fi+"_cfg.py")).read())
  file_name = run_config["cfg_file"]
  file_name = file_name.split("\\")[-1]
  cfgs = file_name.split(".")[-2].split("x")
  self.metrics_data["motor_cfg_0"] = float(cfgs[0])
  self.metrics_data["motor_cfg_1"] = float(cfgs[1])
