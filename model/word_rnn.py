import tensorflow as tf
from tensorflow.contrib import rnn


class WordRNN():
  def __init__(self, vocabulary_size, max_document_length, num_class):
    self.embedding_size = 256
    self.num_hiddens = 512
    self.fc_num_hidden = 256

    self.x = tf.placeholder(tf.int32, [None, max_document_length])
    self.x_len = tf.reduce_sum(tf.sign(self.x), 1)
    self.y = tf.placeholder(tf.int32, [None])
    self.keep_prob = tf.placeholder(tf.float32, [])

