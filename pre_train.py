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