import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
from data import loadData, loadSTL10Labeled
from model import DCGAN, discriminator
from args import parseArgs
from visualization import visualize
import sys
import os.path
import traceback
import progressbar


def train(args):
  print 'Building computational graph..'
  real_images = None
  labels = None
  if args.dataset == 'mnist':
    real_images, labels = loadData(args)
  else:
    real_images = loadData(args)
  z = tf.random_uniform([args.batch_size, args.z_dim], minval=-1, maxval=1, name='z')
  G_loss, D_loss, fake_images, G_summaries, D_summaries = DCGAN(real_images, z, labels, args)
  optimizer_G = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1)
  optimizer_D = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1)
  update_G = slim.learning.create_train_op(G_loss, optimizer_G,
                                           update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='^generator*'),
                                           variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='^generator*'))
  update_D = slim.learning.create_train_op(D_loss, optimizer_D, 
                                          update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='^discriminator*'),
                                          variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='^discriminator*'))
  G_merged_summary = tf.merge_summary(G_summaries)
  D_merged_summary = tf.merge_summary(D_summaries)

  saver = tf.train.Saver()

  print 'Launching session..'
  with tf.Session() as sess:
    print 'Initializing variables..'
    sess.run(tf.initialize_all_variables())

    print 'Staring the queue..'
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print 'Training..'
    writer = tf.train.SummaryWriter(args.log_dir, sess.graph)
    try:
      global_step = 0
      for epoch in xrange(args.n_epochs):
        print 'Epoch #%d' % epoch
        epoch_step = 0
        with progressbar.ProgressBar(0, args.N) as bar:
          while not coord.should_stop():
            epoch_step += 1
            if epoch_step * args.batch_size >= args.N:
              break
            global_step += 1
            # update D
            loss, summary = sess.run([update_D, D_merged_summary])
            writer.add_summary(summary, global_step)
            # update G
            _, summary = sess.run([update_G, G_merged_summary])
            writer.add_summary(summary, global_step) 
            _, summary = sess.run([update_G, G_merged_summary])
            writer.add_summary(summary, global_step)
            bar.update(epoch_step * args.batch_size)

        samples = sess.run(fake_images)
        samples_filename = os.path.join(args.samples_dir, 'samples_%d.png' % epoch)
        visualize(samples, samples_filename)
        print 'samples saved: %s' % samples_filename
        
        checkpoint_filename = os.path.join(args.checkpoint_dir, 'model_%d.ckpt' % epoch)
        saver.save(sess, checkpoint_filename) 
        print 'checkpoint saved: %s' % checkpoint_filename

    except Exception as ex:
      traceback.print_exc(file=sys.stdout)
    finally:
      coord.request_stop()

    coord.join(threads)


def sample(args):
  '''Sample the output of G given a model checkpoint'''
  print 'Building computational graph..'
  z = tf.random_uniform([args.batch_size, args.z_dim], minval=-1, maxval=1, name='z')
  y = tf.one_hot(tf.random_uniform([args.batch_size], minval=0, maxval=args.n_classes-1, dtype=tf.int32), depth=args.n_classes, dtype=tf.float32)
  fake_images = DCGAN(None, z, y, args)

  saver = tf.train.Saver()

  print 'Launching session..'
  with tf.Session() as sess:
    print 'Loading model checkpoint %s..' % args.checkpoint
    saver.restore(sess, args.checkpoint)
    
    res = sess.run(fake_images)
    visualize(res, args.output)
    print 'samples saved: %s' % args.output


def semi(args):
  '''Train on the labeled data in STL10''' 
  print 'Building computational graph..'
  image_batch_train, label_batch_train, image_batch_test, label_batch_test = loadSTL10Labeled(args)
  #discriminator(image_batch_train, None, False, args)
  saver = tf.train.Saver()
  print 'Launching session..' 
  with tf.Session() as sess:
    print 'Initializing..'
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    print 'Loading model checkpoint %s..' % args.checkpoint
    saver.restore(sess. args.checkpoint)
    print 'Staring the queue..'
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
     sess.run([image_batch_train, label_batch_train, image_batch_test, label_batch_test])
    except Exception as ex:
      traceback.print_exc(file=sys.stdout)
    finally:
      coord.request_stop()

    coord.join(threads)


if __name__ == '__main__':

  args = parseArgs()
  print args

  if args.mode == 'train':
    train(args)
  elif args.mode == 'sample':
    sample(args)
  else:
    semi(args)

  print 'Exiting..'
