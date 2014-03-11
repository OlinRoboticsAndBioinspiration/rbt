import numpy as np
import os

def cfg_metrics(self, **kwds):
  di,fi = os.path.split(self.fi)
  run_config = eval(open(os.path.join(di,fi+"_cfg.py")).read())
  file_name = run_config["cfg_file"]
  cfgs = file_name.split(".")[0].split("x")
  self.metrics_data["motor_cfg_0"] = float(cfgs[0])
  self.metrics_data["motor_cfg_1"] = float(cfgs[1])
