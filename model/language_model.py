import tensorflow as tf
from tensorflow.contrib import rnn


class LanguageModel:
    def __init__(self, word_dict, max_document_length):
        self.embedding_size = 256
        self.num_hidden = 512
        self.vocabulary_size = len(word_dict)

        self.x = tf.placeholder(tf.int32, [None, max_document_length])
        self.batch_size = tf.shape(self.x)[0]