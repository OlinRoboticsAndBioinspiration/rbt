import numpy as np
import os

def cfg_metrics(self, **kwds):
  """
  Writes metrics about the robot trial configuration.
  For now, just writing motor configuration thats set in run_config
  Filenames take the form of 10x10 or 10x10@340
  Prerequisites:
  Effects: Sets motor_cfg_0 metric, and motor_cfg_1 metric
  """
  di,fi = os.path.split(self.fi)
  run_config = eval(open(os.path.join(di,fi+"_cfg.py")).read())
  file_name = run_config["cfg_file"]
  file_name = file_name.split("\\")[-1]
  print file_name
  possible_offset = file_name.split("@")
  offset = None
  if len(possible_offset) > 1:
    just_x = possible_offset[0]
    offset = possible_offset[1].split(".")[-2]
  else:
    just_x = possible_offset[0].split(".")[-2]

  cfgs = just_x.split("x")
  self.metrics_data["motor_cfg_0"] = float(cfgs[0])
  self.metrics_data["motor_cfg_1"] = float(cfgs[1])
  if offset is not None:
    self.metrics_data["motor_phase_offset"] = float(offset)
  print self.metrics_data
