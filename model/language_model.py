import tensorflow as tf
from tensorflow.contrib import rnn


class LanguageModel:
    def __init__(self, word_dict, max_document_length):
        self.embedding_size = 256
        self.num_hidden = 512
        self.vocabulary_size = len(word_dict)

        self.x = tf.placeholder(tf.int32, [None, max_document_length])
        self.batch_size = tf.shape(self.x)[0]

        self.lm_input = tf.concat([tf.ones([self.batch_size, 1], tf.int32) * word_dict["<s>"], self.x], axis=1)
        self.lm_output = tf.concat([self.x, tf.ones([self.batch_size, 1], tf.int32) * word_dict["</s>"]], axis=1)
        self.seq_len = tf.reduce_sum(tf.sign(self.lm_input), 1)

        with tf.variable_scope("embedding"):
            init_embeddings = tf.random.uniform([self.vocabulary_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            lm_input_emb = tf.nn.embedding_lookup(embeddings, self.lm_input)

        with tf.variable_scope("rnn"):
            cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.num_hidden)
            rnn_outputs, _ = tf.nn.dynamic_rnn()
            