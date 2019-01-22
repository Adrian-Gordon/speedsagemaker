import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import os

from preprocess_racing_data import preprocess

from linear_regressor_model import *

import tensorflow as tf

#Sample usage run from top-level RacingSpeed directory: 
#import sys
#sys.path.append('LinearRegressor')
#from linear_regressor_learner import *
#
#learner = LinearRegressorLearner(datapath='data/someracesdata.csv',scalerfilename='save/scaler.save', learningrate=0.005, epochs=300,displaystep=10,samplesize=100000,n_features=4,savefilename='save/linearregressor/mv_linear_regressor')
#
#learner.learn()


class LinearRegressorLearner:
  def __init__(self, **kwargs):
    self.data_path = kwargs.get('datapath')
    self.scaler_filename = kwargs.get("scalerfilename")
    #hyperparameters
    self.learning_rate   = kwargs.get("learningrate")
    self.training_epochs = kwargs.get("epochs")
    self.display_step = kwargs.get("displaystep")
    self.sample_size=kwargs.get("samplesize")
    self.n_features = kwargs.get("n_features")
    self.save_filename = kwargs.get("savefilename")


    self.racing_data = pd.read_csv(self.data_path)

    self.racing_data = preprocess(self.racing_data)


    #standardise everything

    scaler = StandardScaler()

    self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']]=scaler.fit_transform(self.racing_data[['speeddiff','distancediff','goingdiff','weightdiff','datediff']])

    #save the scaler for later use

    joblib.dump(scaler, self.scaler_filename)

  def learn(self):

    #instantiate the learning model

    linear_regressor = LinearRegressor(n_features=self.n_features,learning_rate=self.learning_rate)

    init = tf.global_variables_initializer()

    sess = tf.Session()

    sess.run(init)

    for epoch in range(self.training_epochs):

      sample = self.racing_data.sample(n=self.sample_size)

      xs = sample[['distancediff','goingdiff','weightdiff','datediff']]
      ys = sample['speeddiff'].values.reshape(self.sample_size,1)

      _, loss = sess.run([linear_regressor.optimizer, linear_regressor.loss],feed_dict={linear_regressor.X:xs, linear_regressor.Y_:ys})

      if epoch % self.display_step ==0:
        print("Epoch: ", epoch,"Loss:", loss)



    allxs=self.racing_data[['distancediff','goingdiff','weightdiff','datediff']]
    allys=self.racing_data['speeddiff'].values.reshape(self.racing_data.shape[0],1)
    global_loss = sess.run(linear_regressor.loss, feed_dict={linear_regressor.X:allxs,linear_regressor.Y_:allys})
    print("global_loss: ", global_loss)
    print("W:" , sess.run(linear_regressor.W), "b: ", sess.run(linear_regressor.b))

    saver = tf.train.Saver
    save_path = saver().save(sess, os.path.join('./', self.save_filename))
    print("Checkpoint saved at: ", save_path)




