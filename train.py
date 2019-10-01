import tensorflow as tf
import argparse
import os
from model.word_rnn import WordRNN
from data_utils import build_word_dict, build_word_dataset, batch_iter, download_dbpedia


NUM_CLASS = 14
BATCH_SIZE = 64
NUM_EPOCHS = 10
MAX_DOCUMENT_LEN = 100


def train(train_x, train_y, test_x, test_y, vocabulary_size, args):
    with tf.Session() as sess:
        model = WordRNN(vocabulary_size, MAX_DOCUMENT_LEN, NUM_CLASS)

        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        # Summary
        loss_summary = tf.summary.scalar("loss", model.loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load variables from pre-trained model
        if not args.pre_trained == "none":
            pre_trained_variables = [v for v in tf.global_variables()
                                     if (v.name.startswith("embedding") or v.name.startswith("birnn")) and "Adam" not in v.name]
            saver = tf.train.Saver(pre_trained_variables)
            ckpt = tf.train.get_checkpoint_state(os.path.join(args.pre_trained, "model"))
            saver.restore(sess, os.path.join(args.pre_trained, "model"))