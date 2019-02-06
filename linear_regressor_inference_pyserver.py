from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from preprocess_racing_data import preprocess

scalerfilename='save/scaler.save'

weights = np.array([-0.30812508, -0.4418057, -0.04384844, 0.0073124])
bias = 7.8782774e-05

columns=["speed1","datediff","distance1","distance2","going1","going2","weight1","weight2","speed2"]

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



    
   

    #load the scaler

    self.scaler=joblib.load(scalerfilename)

    #standardise everything

    #self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']]=self.scaler.transform(self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']])



    predictions = []
    for _, race in enumerate(self.data):  
      #print(race)
       #data =data = kwargs['data']
      racing_data = pd.DataFrame(race, columns=columns)
      racing_data = preprocess(racing_data)
      racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']]=self.scaler.transform(racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']])

      nparr =racing_data[['distancediff','goingdiff','weightdiff','datediff']].values

      #print(nparr[0])

      scaled_result = np.dot(nparr[0], weights) + bias
      racing_data['speeddiff'] = scaled_result

      #print(self.racing_data)
      

      result = self.scaler.inverse_transform(racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']])

      #inferrer = LinearRegressorInferrer(regressor=linear_regressor,scalerfilename='save/scaler.save',n_features=4,restorefilename='save/mv_linear_regressor')

      #prediction = inferrer.infer(race)
      predictions.append(result[0][0] + racing_data['speed1'][0])

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