import tensorflow as tf
import argparse
import os 
from model.auto_encoder import AutoEncoder
from model.language_model import LanguageModel


BATCH_SIZE = 64
NUM_EPOCHS = 10
MAX_DOCUMENT_LEN = 100


def train(train_x, train_y, word_dict, args):
  with tf.Session() as sess:
    if args.model == "auto_encoder":
      model = AutoEncoder(word_dict, MAX_DOCUMENT_LEN)
    elif args.model == "language_model":
      model = LanguageModel(word_dict, MAX_DOCUMENT_LEN)
    else:
      raise ValueError("Invalid model: {0}. Use auto_encoder | language_model".format(args.model))

    global_step = tf.Variable(0, trainable=False)
    params = tf.trainable_variables()
    gradients = tf.gradients(model.loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

    loss_summary = tf.summary.scalar("loss", model.loss)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.model, sess._graph)

    sess.run(tf.global_variables_initializer())
    