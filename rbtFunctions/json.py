import numpy as np
import json

def json(self, dbg=False, **kwds):
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
  json_file_name = self.fi +  ".json";
  json_file = open(json_file_name, 'wb_')
  json_file.write(json_dumps)
  json_file.close()
