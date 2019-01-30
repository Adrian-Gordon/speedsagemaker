from http.server import BaseHTTPRequestHandler, HTTPServer
import json

#Sample usage: 
import sys
sys.path.append('LinearRegressor')
from linear_regressor_inferrer import *
# dict = { "speed1":[15.2739131669902],"datediff":[816],"distance1":[1613.0016],"distance2":[1566.3672],"going1":[-1],"going2":[-2],"weight1":[133],"weight2":[138],"speed2":[15.658191632928474]}
#
linear_regressor =  LinearRegressor(n_features=4) 
#inferrer = LinearRegressorInferrer(regressor=linear_regressorscalerfilename='save/scaler.save',n_features=4,restorefilename='save/mv_linear_regressor')
#
#inferrer.infer(dict)


class InferenceServerHandler(BaseHTTPRequestHandler):
  def _set_headers(self):
    self.send_response(200)
    self.send_header('Content-type','application/json')
    self.end_headers()

  def do_POST(self):
    if(self.path == '/predict'):
      self.do_inference()
    elif(self.path == '/getmode'):
      self.do_get_mode()

  def do_inference(self):
    self._set_headers()

    self.data_string = self.rfile.read(int(self.headers['Content-Length'])).decode("utf-8")

    print(self.data_string)

    self.data = json.loads(self.data_string)

    predictions = []
    for _, race in enumerate(self.data):  
      #print(race)
      inferrer = LinearRegressorInferrer(regressor=linear_regressor,scalerfilename='save/scaler.save',n_features=4,restorefilename='save/mv_linear_regressor')

      prediction = inferrer.infer(race)
      predictions.append(prediction)

    self.wfile.write(json.dumps(predictions).encode('utf-8'))

  def do_get_mode(self):
    self._set_headers()

    self.data_string = self.rfile.read(int(self.headers['Content-Length'])).decode("utf-8")

   # print(self.data_string)

    self.data = json.loads(self.data_string)
    mp = ModeProcessor()
    mode = mp.getMode(self.data)
    self.wfile.write(str(mode).encode('utf-8'))


class ModeProcessor:
  

  def getMode(self, arr):
    n_bins = 10
    found_mode = False
    np_arr = np.array(arr)

    while n_bins > 0 and found_mode == False:

      count, edges = np.histogram(arr, bins = n_bins)
      #print(count, " " , edges)

      has_mode, mode_index = self.hasMode(count)
      if(has_mode):
        #print("mode: ", mode_index)
        lower_bound = edges[mode_index]
        upper_bound = edges[mode_index + 1]

        res = np.mean(np_arr[np.logical_and(np_arr >= lower_bound, np_arr<= upper_bound)])

        #print("res: " ,res)
        return res

      else:
        n_bins = n_bins - 1

  def hasMode(self, arr):
    max_count = -1
    index = -1
    it = np.nditer(arr, flags=['f_index'])
    while not it.finished:
      #print(it.index," ",it[0])
      if it[0] > max_count:
        max_count = it[0]
        index = it.index
      elif it[0] == max_count: #multi modes
        return False, -1
      it.iternext()
    if it.finished:
      return True, index


#mp = ModeProcessor()

#x = mp.getMode([8.0,11.0, 14.5,12.6, 18.7,12.8, 14.0, 13.0, 18.0,8.6])

#print(x)
#mp.getMode([11.0])

#m = mp.hasMode(np.array([1,1]))
#print(m)




if __name__ == '__main__':
  server = HTTPServer(('',3002), InferenceServerHandler)
  print("listening on port 3002")
  server.serve_forever()