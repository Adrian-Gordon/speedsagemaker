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
#from dnn_regressor_learner import *
#
#learner = DNNRegressorLearner(datapath='data/someracesdata.csv',scalerfilename='save/scaler.save', hidden_units=[10, 10, 10],batchsize=100000)
#
#learner.learn(100) #100 steps

class DNNRegressorLearner:
  def __init__(self, **kwargs):
    self.data_path = kwargs.get('datapath')
    self.scaler_filename = kwargs.get("scalerfilename")
    #hyperparameters

   # self.training_epochs = kwargs.get("epochs")
    self.batch_size=kwargs.get("batchsize")
    self.hidden_units = kwargs.get("hidden_units")
   # self.save_filename = kwargs.get("savefilename")


    self.racing_data = pd.read_csv(self.data_path)

    self.racing_data = preprocess(self.racing_data)


    #standardise everything

    scaler = StandardScaler()

    self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']]=scaler.fit_transform(self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']])

    #save the scaler for later use

    joblib.dump(scaler, self.scaler_filename)

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

    
  def learn(self, steps):
    self.regressor.train(self.np_training_input_fn(self.x_dict, self.speeddiff, self.batch_size),steps=steps)  

  def np_training_input_fn(self, x, y, batch_size):
    return tf.estimator.inputs.numpy_input_fn(
      x = x,
      y = y,
      batch_size = batch_size,
      num_epochs = None,
      shuffle = True,
      queue_capacity = 5000)

