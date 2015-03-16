import numpy as np
import json
import os

def json_task(self, dbg=False, **kwds):
  """
  Dump ukf data to json file

  Prerequisites: {ukf | load}
  Effects:
    Creates json file and saves to disk.
  """
  json_dir = "json"
  x_pos = self.X[..., self.j['x']]
  y_pos = self.X[..., self.j['y']]
  z_pos = self.X[..., self.j['z']]
  yaw = self.X[..., self.j['yaw']]
  pitch = self.X[..., self.j['pitch']]
  roll = self.X[..., self.j['roll']]
  obj = {
          "time": self.t.tolist(),
          "ukf": {
              'x':x_pos.tolist(),
              'y':y_pos.tolist(),
              'z':z_pos.tolist(),
              'yaw':yaw.tolist(),
              'pitch':pitch.tolist(),
              'roll':roll.tolist()
              },
          "center":[np.mean(x_pos), np.mean(y_pos)]
          }

  json_dumps = json.dumps(obj);

  folder = "/".join(self.fi.split("/")[0:-1])
  if not os.path.exists(os.path.dirname(self.fi) + "/" + json_dir):
    os.mkdir(os.path.dirname(self.fi) + "/" + json_dir)

  json_file_name = os.path.dirname(self.fi) + "/" + json_dir + "/" + os.path.basename(self.fi) +  ".json";
  json_file = open(json_file_name, 'wb+')

  json_file.write(json_dumps)
  json_file.close()
