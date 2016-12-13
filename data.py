import tensorflow as tf
import os.path
import os
import cPickle
from glob import glob
import struct
import numpy as np
from scipy.misc import imsave
import progressbar


def parseIdxFormat(byte_array):
  '''Parse the raw MNIST data format'''
  assert byte_array[0] == byte_array[1] == '\x00'
  
  dtype = None
  if byte_array[2] == '\x08':
    dtype = np.uint8
  elif byte_array[2] == '\x09':
    dtype = np.int8
  elif byte_array[2] == '\x0B':
    dtype = np.int16
  elif byte_array[2] == '\x0C':
    dtype = np.int32
  elif byte_array[2] == '\x0D':
    dtype = np.float32
  else:
    dtype = np.float64
  
  n_dim = struct.unpack('>B', byte_array[3])[0]
  shape = [0] * n_dim
  for i in xrange(n_dim):
    shape[i] = struct.unpack('>i', byte_array[4 + 4 * i: 8 + 4 * i])[0]
  
  return np.frombuffer(byte_array, dtype, -1, 4 + 4 * n_dim).reshape(shape)


def parseLabelFile(byte_array):
  '''Parse the raw MNSIT label format'''
  assert byte_array[0] == byte_array[1] == '\x00'
  
  dtype = None 
  if byte_array[2] == '\x08': 
    dtype = np.uint8 
  elif byte_array[2] == '\x09': 
    dtype = np.int8 
  elif byte_array[2] == '\x0B': 
    dtype = np.int16 
  elif byte_array[2] == '\x0C': 
    dtype = np.int32 
  elif byte_array[2] == '\x0D': 
    dtype = np.float32 
  else: 
    dtype = np.float64

  assert byte_array[3] == '\x01'

  return np.frombuffer(byte_array, dtype, -1, 8).flatten()


def prepareImages(args):
  '''Download and store the data in appropriate format'''
  # create the folders for storing the images
  if not os.path.exists('./data'):
    os.mkdir('./data')
  root_dir = os.path.join('./data', args.dataset)
  if not os.path.exists(root_dir):
    os.mkdir(root_dir)
  if args.dataset == 'stl10':
    if not os.path.exists(os.path.join(root_dir, 'unlabeled')):
      os.mkdir(os.path.join(root_dir, 'unlabeled'))
    if not os.path.exists(os.path.join(root_dir, 'train')):
      os.mkdir(os.path.join(root_dir, 'train'))
    if not os.path.exists(os.path.join(root_dir, 'test')):
      os.mkdir(os.path.join(root_dir, 'test'))
  

  if args.dataset == 'stl10':
    # download and extract the dataset
    if not os.path.exists('stl10_binary.tar.gz'):
      print 'Downloading stl10..'
      os.system('wget -c http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz')
    if not os.path.exists('stl10_binary'):
      print 'Extracting stl10_binary.tar.gz..'
      os.system('tar -xf stl10_binary.tar.gz')

    # save unlabeled images
    if len(glob(os.path.join(root_dir, 'unlabeled', '*.jpg'))) < 100000:
      print 'Reading unlabeled images..'
      bytes_unlabeled = open('./stl10_binary/unlabeled_X.bin', 'rb').read()
      img_size = 3 * 96 * 96
      n_images = len(bytes_unlabeled) / img_size
      with progressbar.ProgressBar(0, n_images) as bar:
        for i in xrange(n_images):
          img = np.frombuffer(bytes_unlabeled, dtype=np.uint8, count=img_size, offset=i * img_size).reshape([3, 96, 96]).transpose([2, 1, 0])
          imsave(os.path.join(root_dir, 'unlabeled', '%05d.jpg' % i), img)
          bar.update(i)
      print '%d Unlabeled images saved to %s' % (n_images, os.path.join(root_dir, 'unlabeled', '*.jpg'))
  
    # save labeled images
    if not os.path.exists(os.path.join(root_dir, 'train', 'X.cPickle')):
      print 'Reading training images..'
      bytes_train_data = open('./stl10_binary/train_X.bin', 'rb').read()
      bytes_train_label = open('./stl10_binary/train_y.bin', 'rb').read()
      img_size = 3 * 96 * 96
      N_train = len(bytes_train_data) / img_size 
      X_train = np.zeros((N_train, 96, 96, 3), dtype=np.uint8)
      y_train = np.zeros((N_train, ), dtype=np.uint8)
      with progressbar.ProgressBar(0, N_train) as bar:
        for i in xrange(N_train):
          X_train[i] = np.frombuffer(bytes_train_data, dtype=np.uint8, count=img_size, offset=i * img_size).reshape([3, 96, 96]).transpose([2, 1, 0])
          y_train[i] = np.frombuffer(bytes_train_label, dtype=np.uint8, count=1, offset=i)
          bar.update(i)
      cPickle.dump(X_train, open(os.path.join(root_dir, 'train', 'X.cPickle'), 'wb'))
      cPickle.dump(y_train, open(os.path.join(root_dir, 'train', 'y.cPickle'), 'wb'))

      print 'Reading testing images..'
      bytes_test_data = open('./stl10_binary/test_X.bin', 'rb').read()
      bytes_test_label = open('./stl10_binary/test_y.bin', 'rb').read()
      N_test = len(bytes_test_data) / img_size
      X_test = np.zeros((N_test, 96, 96, 3), dtype=np.uint8)
      y_test = np.zeros((N_test, ), dtype=np.uint8)
      with progressbar.ProgressBar(0, N_test) as bar:
        for i in xrange(N_test):
          X_test[i] = np.frombuffer(bytes_test_data, dtype=np.uint8, count=img_size, offset=i * img_size).reshape([3, 96, 96]).transpose([2, 1, 0])
          y_test[i] = np.frombuffer(bytes_test_label, dtype=np.uint8, count=1, offset=i)
          bar.update(i)
      cPickle.dump(X_test, open(os.path.join(root_dir, 'test', 'X.cPickle'), 'wb'))
      cPickle.dump(y_test, open(os.path.join(root_dir, 'test', 'y.cPickle'), 'wb'))

  else:
    # download and extract the dataset
    if not os.path.exists('train-images-idx3-ubyte.gz'):
      print 'Downloading mnist..'
      os.system('wget -c http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
      os.system('wget -c http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
      os.system('wget -c http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
      os.system('wget -c http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
    if not os.path.exists('train-images-idx3-ubyte'):
      os.system('gunzip train-images-idx3-ubyte.gz')
    if not os.path.exists('t10k-images-idx3-ubyte'):
      os.system('gunzip t10k-images-idx3-ubyte.gz')
    if not os.path.exists('train-labels-idx1-ubyte'):
      os.system('gunzip train-labels-idx1-ubyte.gz')
    if not os.path.exists('t10k-labels-idx1-ubyte'):
      os.system('gunzip t10k-labels-idx1-ubyte.gz')

    # saving images
    if not os.path.exists(os.path.join(root_dir, 'X.cPickle')):
      X_train = parseIdxFormat(open('./train-images-idx3-ubyte', 'rb').read())
      y_train = parseLabelFile(open('./train-labels-idx1-ubyte', 'rb').read())
      X_test = parseIdxFormat(open('t10k-images-idx3-ubyte', 'rb').read())
      y_test = parseLabelFile(open('t10k-labels-idx1-ubyte', 'rb').read())
      X = np.concatenate((X_train, X_test), axis=0)
      y = np.concatenate((y_train, y_test), axis=0)
      cPickle.dump(X.astype(np.uint8), open(os.path.join(root_dir, 'X.cPickle'), 'wb'))
      cPickle.dump(y.astype(np.uint8), open(os.path.join(root_dir, 'y.cPickle'), 'wb'))
      print '%d images and labels saved' % X.shape[0]  


def loadSTL10Labeled(args):
  '''Load the labeled data in STL-10 for semi-supervised learning'''
  prepareImages(args)
  data_dir = os.path.join('./data', 'stl10')
  
  with tf.name_scope('InputPipeline'):
    # read images
    print 'Reading data from disk..'
    X_train = cPickle.load(open(os.path.join(data_dir, 'train', 'X.cPickle'), 'rb'))
    y_train = cPickle.load(open(os.path.join(data_dir, 'train', 'y.cPickle'), 'rb'))
    X_test = cPickle.load(open(os.path.join(data_dir, 'test', 'X.cPickle'), 'rb'))
    y_test = cPickle.load(open(os.path.join(data_dir, 'test', 'y.cPickle'), 'rb'))
    raw_image_train, label_train = tf.train.slice_input_producer([X_train, y_train], capacity=4 * args.batch_size)
    raw_image_test, label_test = tf.train.slice_input_producer([X_test, y_test], num_epochs=1, capacity=4 * args.batch_size)
    # preprocessing
    cropped_image_train = tf.random_crop(raw_image_train, [args.img_crop_height, args.img_crop_width, args.img_depth], name='cropped_image_train')
    cropped_image_test = tf.image.resize_image_with_crop_or_pad(raw_image_test, args.img_crop_height, args.img_crop_width)
    image_train = tf.image.resize_images(cropped_image_train, [args.output_height, args.output_width])
    image_test = tf.image.resize_images(cropped_image_test, [args.output_height, args.output_width])
    # generate batches
    image_batch_train, label_batch_train = tf.train.shuffle_batch([image_train, label_train], args.batch_size, 
                                                                  capacity=4 * args.batch_size, 
                                                                  min_after_dequeue=args.batch_size, 
                                                                  num_threads=args.n_threads, name='image_batch_train')
    image_batch_test, label_batch_test = tf.train.batch([image_test, label_test], args.batch_size, 
                                                        capacity=4 * args.batch_size, num_threads=args.n_threads, 
                                                        name='image_batch_test')     
  return (image_batch_train, label_batch_train, image_batch_test, label_batch_test)
   

def loadData(args):
  '''Load data in MNIST or STL10 to train DCGANs'''
  # prepare image files
  prepareImages(args)
  data_dir = os.path.join('./data', args.dataset)

  with tf.name_scope('InputPipeline'):
    if args.dataset == 'mnist':
      print 'Reading data from disk..'
      X = cPickle.load(open(os.path.join(data_dir, 'X.cPickle'), 'rb'))
      y = cPickle.load(open(os.path.join(data_dir, 'y.cPickle'), 'rb'))
      raw_image, label = tf.train.slice_input_producer([X, y], capacity=4 * args.batch_size)
      image = tf.image.resize_images(tf.reshape(raw_image, [args.img_crop_height, args.img_crop_width, args.img_depth]), [args.output_height, args.output_width])
      # generate batches
      image_batch, label_batch = tf.train.shuffle_batch([image, label], args.batch_size, capacity=4 * args.batch_size, 
                                                        min_after_dequeue=args.batch_size, num_threads=args.n_threads, 
                                                        name='image_batch')  
      return tf.cast(image_batch, tf.float32), tf.one_hot(label_batch, depth=args.n_classes)
    
    else:
      # glob image files
      filenames = tf.train.match_filenames_once(os.path.join(data_dir, 'unlabeled', '*.jpg'), name='filenames')
      filename_queue = tf.train.string_input_producer(filenames, capacity=5 * args.batch_size, name='filename_queue')

      # read images
      img_reader = tf.WholeFileReader()
      _, value = img_reader.read(filename_queue)
      raw_image = tf.cast(tf.image.decode_jpeg(value), tf.float32, name='raw_image')                        
      
      # preprocessing
      cropped_image = tf.random_crop(raw_image, [args.img_crop_height, args.img_crop_width, args.img_depth], name='cropped_image')
      image = tf.image.resize_images(cropped_image, [args.output_height, args.output_width])

      # generate batches
      image_batch = tf.train.shuffle_batch([image], args.batch_size, capacity=4 * args.batch_size, 
                                           min_after_dequeue=args.batch_size, num_threads=args.n_threads, 
                                           name='image_batch')
      return tf.cast(image_batch, tf.float32)


if __name__ == '__main__':
  import traceback
  import sys
  from args import parseArgs
  args = parseArgs()
  image_batch = None
  label_batch = None

  if args.dataset == 'mnist':
    image_batch, label_batch = loadData(args)
  else:
    image_batch = loadData(args)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      print 'Trying to load 100 data batches'
      for i in xrange(100):
        img = sess.run(image_batch)
        print img.shape
      print 'Done'
    except Exception as ex:
      traceback.print_exc(file=sys.stdout)
    finally:
      coord.request_stop()

    coord.join(threads)
