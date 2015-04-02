import os

vis_dir = "vis"

def vis3d(self, dbg=False, **kwds):
  """
  Used for visualization of raw robot run
  Prerequisites: {ukf | load} json
  Effects: Writes a html file with webgl visualization
  """
  di,fi = os.path.split(self.fi)
  di = os.path.join(di, vis_dir)
  if not os.path.exists( di ):
    os.mkdir( di )
  print fi, "FFI"
  current_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
  template_loc = os.path.join(current_location, "template.html")
  template = open(template_loc, 'r+').read()
  # TODO make this string manupulation actually correct
  json_fi= "".join(self.fi.split("/")[0:-1]) + "/json/" + self.fi.split("/")[-1]
  json = open(json_fi+ ".json", 'r+').read()
  html_out = open(os.path.join(di, fi + "_vis.html"), 'wb+')
  template = template.replace("JSON_DATA_REPLACE", json)
  html_out.write(template)
  html_out.close()
