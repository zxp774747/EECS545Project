import os, sys, tarfile, urllib
import numpy as np
import matplotlib.pyplot as plt
import math
from stl10_train import DCGAN
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "Tsize of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


HEIGHT = 96
WIDTH = 96
DEPTH = 3
OUTPUT_SIZE = 64
SIZE = HEIGHT * WIDTH * DEPTH

DATA_DIR = './data'
TRAIN_DATA_PATH = './data/stl10_binary/train_X.bin'
TRAIN_LABEL_PATH = './data/stl10_binary/train_y.bin'
TEST_DATA_PATH = './data/stl10_binary/test_X.bin'
TEST_LABEL_PATH = './data/stl10_binary/test_y.bin'


def read_data(path_to_data):

    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
	data = np.reshape(everything, (-1, 3, 96, 96))

	return data


def read_labels(path_to_labels):

    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def init(labels):
	label_vec = np.zeros((len(labels), 10), dtype=np.float)	
	for i in range(len(labels)):
		label_vec[i, labels[i] - 1] = 1.0
	return label_vec

def print_final_image(data):	
	length = len(data)
	batch = int(math.floor(length/OUTPUT_SIZE))
	for i in range(batch):
		start = i*OUTPUT_SIZE
		end = (i+1)*OUTPUT_SIZE
		row = int(math.sqrt(OUTPUT_SIZE))
		final_image = []
		for j in range(row):
			row_image = []
			for k in range(row):
				current_image =  np.transpose(data[start + j*row + k])
				if len(row_image) == 0:
					row_image = current_image
				else:
					row_image = np.concatenate((row_image,current_image),axis = 1)		
			if len(final_image) == 0:	
				final_image = row_image
			else:
				final_image = np.concatenate((final_image,row_image),axis = 0)
		plt.imshow(final_image)
    		plt.show()

def shuffle(data, label, seed):
	np.random.seed(seed)
        np.random.shuffle(data)
        np.random.seed(seed)
        np.random.shuffle(label)

if __name__ == "__main__":
	train_data = read_data(TRAIN_DATA_PATH)
	test_data = read_data(TEST_DATA_PATH)
	train_labels = read_labels(TRAIN_LABEL_PATH)
	test_labels = read_labels(TEST_LABEL_PATH)

	label_result = init(train_labels)
	
	test_data_subset = test_data[:64]
	test_labels_subset = test_labels[:64]
	shuffle(test_data_subset, test_labels_subset, 100)
	a = np.transpose(test_data_subset, (0,3,2,1))
	print a.shape
	print test_data_subset.shape
        with tf.Session() as sess:
		dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, y_dim=10, output_size=96, c_dim=3, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir)
		dcgan.train(FLAGS, a, label_result)

	#print_final_image(test_data_subset)
	#print test_labels_subset

