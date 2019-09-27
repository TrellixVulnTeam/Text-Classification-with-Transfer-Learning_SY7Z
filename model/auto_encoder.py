import tensorflow as tf
from tensorflow.contrib import rnn


class AutoEncoder(object):
    def __init__(self, word_dict, max_document_length):
        self.embedding_size = 256
        self.num_hidden = 512
        self.vocabulary_size = len(word_dict)

        self.x = tf.placeholder(tf.int32, [None, max_document_length])
        self.batch_size = tf.shape(self.x)[0]

        self.decoder_input = tf.concat([tf.ones([self.batch_size, 1], tf.int32) * word_dict["<s>"], self.x], axis=1)
        self.decoder_output = tf.concat([self.x, tf.ones([self.batch_size, 1], tf.int32) * word_dict["</s>"]], axis=1)
        self.encoder_input_len = tf.reduce_sum(tf.sign(self.x), 1)
        self.decoder_input_len = tf.reduce_sum(tf.sign(self.decoder_input), 1)

        with tf.variable_scope("embedding"):
            init_embeddings = tf.random.uniform([self.vocabulary_size, self.embedding_size])
            embeddigs = tf.get_variable("embeddings", initializer=init_embeddings)
            encoder_input_emb = tf.nn.embedding_lookup(embeddigs, self.x)
            decoder_input_emb = tf.nn.embedding_lookup(embeddigs, self.decoder_input)
        
        with tf.variable_scope("rnn"):
            encoder_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.num_hidden)

            _, encoder_states = tf.nn.dynamic_rnn(encoder_cell,
                                                  encoder_input_emb,
                                                  sequence_length=self.encoder_input_len,
                                                  dtype=tf.float32)
        
        with tf.variable_scope("decoder"):
            decoder_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.num_hidden)

            decoder_outputs, _ = tf.nn.dynamic_rnn(decoder_cell,
                                                   decoder_input_emb,
                                                   sequence_length=self.decoder_input_len,
                                                   initial_state=encoder_states,
                                                   dtype=tf.float32)

        