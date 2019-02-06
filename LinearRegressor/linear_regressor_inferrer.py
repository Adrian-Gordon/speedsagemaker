import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import os

from linear_regressor_model import *

import tensorflow as tf

from preprocess_racing_data import preprocess

#Sample usage: 
#import sys
#sys.path.append('LinearRegressor')
#from linear_regressor_inferrer import *
# dict = { "speed1":[15.2739131669902],"datediff":[816],"distance1":[1613.0016],"distance2":[1566.3672],"going1":[-1],"going2":[-2],"weight1":[133],"weight2":[138],"speed2":[15.658191632928474]}
#
#linear_regressor =  LinearRegressor(n_features=4) 
#inferrer = LinearRegressorInferrer(regressor=linear_regressor,scalerfilename='save/scaler.save',n_features=4,restorefilename='save/mv_linear_regressor')
#
#inferrer.infer(dict)

class LinearRegressorInferrer:
  def __init__(self, **kwargs):
    self.scaler_filename = kwargs.get("scalerfilename")
    self.n_features = kwargs.get("n_features")
    self.restore_filename = kwargs.get("restorefilename")
    self.linear_regressor = kwargs.get("regressor")

  def infer(self,data):
    columns=["speed1","datediff","distance1","distance2","going1","going2","weight1","weight2","speed2"]
    
    #data =data = kwargs['data']
    self.racing_data = pd.DataFrame(data, columns=columns)
    self.racing_data = preprocess(self.racing_data)

    #load the scaler

    self.scaler=joblib.load(self.scaler_filename)

    #standardise everything

    self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']]=self.scaler.transform(self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']])



    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)
      saver = tf.train.Saver
      saver().restore(sess, os.path.join('./',self.restore_filename))
     
      allxs=self.racing_data[['distancediff','goingdiff','weightdiff','datediff']]

      yVal = sess.run(self.linear_regressor.Y,feed_dict ={self.linear_regressor.X:allxs})

      #print("predictedValue: ", yVal)

      self.racing_data['speeddiff'] = pd.Series(yVal[0])

      #print(self.racing_data)
      

      result = self.scaler.inverse_transform(self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']])

      print("W:" , sess.run(self.linear_regressor.W), "b: ", sess.run(self.linear_regressor.b))
      #print("result: ", result)

      return((result[0,0] + self.racing_data['speed1'])[0])

    