import tensorflow as tf

import functools


def define_scope(function):
  attribute ='_cache_' + function.__name__

  @property
  @functools.wraps(function)
  def decorator(self):
    if not hasattr(self, attribute):
      with tf.variable_scope(function.__name__):
        setattr(self, attribute, function(self))
    return getattr(self, attribute)
  return decorator



class LinearRegressor:
  def __init__(self, **kwargs):

    n_features = kwargs.get("n_features")
    self.learning_rate =kwargs.get('learning_rate',0.01)
    self.X = tf.placeholder(tf.float32, [None,n_features],name="x")
    self.Y_ = tf.placeholder(tf.float32, [None,1],name="speeddiff")
    self.W = tf.Variable(tf.zeros([n_features,1]), name="weight")
    self.b = tf.Variable(tf.zeros([1]), name="bias")

    self.Y = tf.matmul(self.X,self.W) + self.b

    self.loss

    self.optimizer


  @define_scope
  def loss(self):
    with tf.variable_scope('Loss'):
     
      loss = tf.reduce_mean(tf.pow(self.Y - self.Y_, 2))

      return loss

  @define_scope
  def optimizer(self):
    return(tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss))
