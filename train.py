import tensorflow as tf
import argparse
import os
from data_utils import build_word_dataset, build_word_dict, batch_iter, download_dbpedia


NUM_CLASS = 14
BATCH_SIZE = 64
NUM_EPOCHS = 10
MAX_DOCUMENT_LEN = 100