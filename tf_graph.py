'''
Load pretrain models and create a tensorflow session to run them

'''
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class FaceRecGraph(object):
    def __init__(self):
        '''
            There'll be more to come in this class
        '''
        self.graph = tf.Graph();
