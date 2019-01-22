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
#from linear_regressor_tester import *
#
#tester = LinearRegressorTester(datapath='data/sometestraces.csv',scalerfilename='save/scaler.save',n_features=4,restorefilename='save/mv_linear_regressor')
#
#tester.test()

class LinearRegressorTester:
  def __init__(self, **kwargs):
    self.data_path = kwargs.get('datapath')
    self.scaler_filename = kwargs.get("scalerfilename")
    self.racing_data = pd.read_csv(self.data_path)
    self.racing_data = preprocess(self.racing_data)
    self.n_features = kwargs.get("n_features")
    self.restore_filename = kwargs.get("restorefilename")

    #load the scaler

    scaler=joblib.load(self.scaler_filename)

    #standardise everything

    self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']]=scaler.transform(self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']])

  def test(self):

    #instantiate the learning model

    linear_regressor = LinearRegressor(n_features=self.n_features)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)
      saver = tf.train.Saver
      saver().restore(sess, os.path.join('./',self.restore_filename))
      allxs=self.racing_data[['distancediff','goingdiff','weightdiff','datediff']]
      allys=self.racing_data['speeddiff'].values.reshape(self.racing_data.shape[0],1)
      global_loss = sess.run(linear_regressor.loss, feed_dict={linear_regressor.X:allxs,linear_regressor.Y_:allys})
      print("global_loss: ", global_loss)
      print("W:" , sess.run(linear_regressor.W), "b: ", sess.run(linear_regressor.b))



