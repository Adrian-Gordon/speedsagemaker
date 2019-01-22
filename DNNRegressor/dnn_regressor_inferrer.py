import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import os

from preprocess_racing_data import preprocess


import tensorflow as tf

#Sample usage run from top-level RacingSpeed directory: 
#import sys
#sys.path.append('DNNRegressor')
#from dnn_regressor_inferrer import *
# dict = { 'speed1':[15.2739131669902],'datediff':[816],'distance1':[1613.0016],'distance2':[1566.3672],'going1':[-1],'going2':[-2],'weight1':[133],'weight2':[138],'speed2':[15.658191632928474]}

#inferrer = DNNRegressorInferrer(data=dict,scalerfilename='save/scaler.save', hidden_units=[10, 10, 10])
#
#inferrer.infer() 

class DNNRegressorInferrer:
  def __init__(self, **kwargs):

    self.scaler_filename = kwargs.get("scalerfilename")
    #hyperparameters

    self.hidden_units = kwargs.get("hidden_units")
   # self.save_filename = kwargs.get("savefilename")


    columns=["speed1","datediff","distance1","distance2","going1","going2","weight1","weight2","speed2"]
    
    data =data = kwargs['data']
    self.racing_data = pd.DataFrame(data, columns=columns)
    self.racing_data = preprocess(self.racing_data)

    #load the scaler

    self.scaler=joblib.load(self.scaler_filename)
    #standardise everything


    self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']]=self.scaler.transform(self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']])

    distancediff = self.racing_data['distancediff'].values
    goingdiff = self.racing_data['goingdiff'].values
    weightdiff = self.racing_data['weightdiff'].values
    datediff = self.racing_data['datediff'].values

    self.speeddiff = self.racing_data['speeddiff'].values

    self.x_dict ={
      'distancediff':distancediff,
      'goingdiff':goingdiff,
      'weightdiff':weightdiff,
      'datediff':datediff
    }

    self.feature_columns = [tf.feature_column.numeric_column(k) for k in self.x_dict.keys()]


    self.regressor = tf.estimator.DNNRegressor(feature_columns = self.feature_columns, hidden_units=self.hidden_units, model_dir = os.getcwd() + "/save/dnnregressor")

    
  def infer(self):
    predicted_vals=[]
    res =self.regressor.predict(self.np_testing_input_fn(self.x_dict, self.speeddiff))  
    for pred in res:
      predicted_vals.append(pred['predictions'])
    self.racing_data['speeddiff'] = predicted_vals[0]
    result = self.scaler.inverse_transform(self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']])
    self.racing_data["speed2"] = self.racing_data["speed1"] + self.racing_data["speeddiff"]
    #print("result: ", result)
    return(self.racing_data)
    # return((result[0,0] + self.racing_data['speed1'])[0])

  def np_testing_input_fn(self, x, y):
    return tf.estimator.inputs.numpy_input_fn(
      x = x,
      y = y,
      batch_size = 1,
      num_epochs = 1,
      shuffle = False)

