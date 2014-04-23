import os
import datetime

def time_metric(self, **kwds):
    di,fi = os.path.split(self.fi)
    time = datetime.datetime.strptime(fi, "%Y%m%d-%H%M")
    seconds = (time - datetime.datetime(1970,1,1)).total_seconds()
    self.metrics_data["eponch"] = seconds
