import tensorflow as tf
import os.path
import os
from glob import glob
import struct
import numpy as np
from scipy.misc import imsave
import progressbar


def parseIdxFormat(byte_array):
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


def prepareImages(dataset):
  # create the folders for storing the images
  if not os.path.exists('./data'):
    os.mkdir('./data')
  root_dir = os.path.join('./data', dataset)
  if not os.path.exists(root_dir):
    os.mkdir(root_dir)
  if not os.path.exists(os.path.join(root_dir, 'unlabeled')):
    os.mkdir(os.path.join(root_dir, 'unlabeled'))

  if dataset == 'stl10':
    # download and extract the dataset
    if not os.path.exists('stl10_binary.tar.gz'):
      print 'Downloading stl10..'
      os.system('wget -c http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz')
    if not os.path.exists('stl10_binary'):
      print 'Extracting stl10_binary.tar.gz..'
      os.system('tar -xf stl10_binary.tar.gz')

    # saving images
    if len(glob(os.path.join(root_dir, 'unlabeled', '*.jpg'))) < 100000:
      print 'Saving images..'
      bytes_unlabeled = open('./stl10_binary/unlabeled_X.bin', 'rb').read()
      img_size = 3 * 96 * 96
      n_images = len(bytes_unlabeled) / img_size
      with progressbar.ProgressBar(0, n_images) as bar:
        for i in xrange(n_images):
          img = np.frombuffer(bytes_unlabeled, dtype=np.uint8, count=img_size, offset=i * img_size).reshape([3, 96, 96]).transpose([2, 1, 0])
          imsave(os.path.join(root_dir, 'unlabeled', '%05d.jpg' % i), img)
          bar.update(i)
      print '%d Unlabeled images saved to %s' % (n_images, os.path.join(root_dir, 'unlabeled', '*.jpg'))
  
  else:
    # download and extract the dataset
    if not os.path.exists('train-images-idx3-ubyte.gz'):
      print 'Downloading mnist..'
      os.system('wget -c http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
      os.system('wget -c http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
    if not os.path.exists('train-images-idx3-ubyte'):
      os.system('gunzip train-images-idx3-ubyte.gz')
    if not os.path.exists('t10k-images-idx3-ubyte'):
      os.system('gunzip t10k-images-idx3-ubyte.gz')

    # saving images
    if len(glob(os.path.join(root_dir, 'unlabeled', '*.jpg'))) < 60000:
      print len(glob(os.path.join(root_dir, 'unlabeled', '*.jpg')))
      bytes_train = open('./train-images-idx3-ubyte', 'rb').read()
      X_train = parseIdxFormat(bytes_train)
      with progressbar.ProgressBar(0, X_train.shape[0]) as bar:
        for i in xrange(X_train.shape[0]):
          imsave(os.path.join(root_dir, 'unlabeled', '%05d.jpg' % i), X_train[i])
          bar.update(i)
      bytes_test = open('t10k-images-idx3-ubyte', 'rb').read()
      X_test = parseIdxFormat(bytes_test)
      with progressbar.ProgressBar(0, X_test.shape[0]) as bar:
        for i in xrange(X_test.shape[0]):
          imsave(os.path.join(root_dir, 'unlabeled', '%05d.jpg' % (i + X_train.shape[0])), X_test[i])
          bar.update(i)
      print '%d images saved to %s' % (X_train.shape[0] + X_test.shape[0], os.path.join(root_dir, 'unlabeled', '*.jpg')) 


def loadData(args):
  # prepare image files
  prepareImages(args.dataset)
  images_dir = os.path.join('./data', args.dataset, 'unlabeled')

  with tf.name_scope('InputPipeline'):
    # glob image files
    filenames = tf.train.match_filenames_once(os.path.join(images_dir, '*.jpg'), name='filenames')
    filename_queue = tf.train.string_input_producer(filenames, capacity=5 * args.batch_size, name='filename_queue')
  
    # read images
    img_reader = tf.WholeFileReader()
    _, value = img_reader.read(filename_queue)
    raw_image = tf.cast(tf.image.decode_jpeg(value), tf.float32, name='raw_image')

    # preprocessing
    if args.dataset == 'stl10':
      cropped_image = tf.random_crop(raw_image, [args.img_height, args.img_width, args.img_depth], name='cropped_image')
      image = cropped_image
    else:
      image = tf.image.resize_images(tf.reshape(raw_image, [28, 28, 1]), [args.img_height, args.img_width])

    # generate batches
    image_batch = tf.train.shuffle_batch([image], args.batch_size, capacity=4 * args.batch_size, 
                                         min_after_dequeue=args.batch_size, num_threads=args.n_threads, 
                                         name='image_batch')
    real_image_summary = tf.image_summary('real_images', image_batch, max_images=4)
                                    
    return image_batch, real_image_summary


if __name__ == '__main__':
  import traceback
  import sys
  from args import parseArgs
  args = parseArgs()
  image_batch = loadData(args)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      print 'Trying to load 100 data batches'
      for i in xrange(100):
        res = sess.run(image_batch)
        print res.shape
      print 'Done'
    except Exception as ex:
      traceback.print_exc(file=sys.stdout)
    finally:
      coord.request_stop()

    coord.join(threads)
