import os
import numpy as np
from util import files,geom,num,util

cdir = 'cal'
ddir = 'dat'

def cal(self,cdir=cdir,**kwds):
  """
  Apply calibration to data

  Inputs:
    (optional)
    cdir - str - directory containing calibration data

  Effects:
    - applies calibration to self.d
  """
  # unpack data
  d0 = self.d; fi = self.fi;
  di,fi = os.path.split(self.fi)
  # load calibration
  c = util.Struct()
  c.read(files.file(fi,di=os.path.join(cdir,ddir),sfx='_cal.py'),
         locals={'array':np.array})
  R,t = c.R,c.t
  # apply calibration to data
  # NOTE: broadcasts over matrix multiplication AND vector addition . . .
  d = np.dot(d0, R.T) - t
  # pack data
  self.d = d
