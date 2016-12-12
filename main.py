import tensorflow as tf
import tensorflow.contrib.slim as slim
from data import loadData
from model import DCGAN
from args import parseArgs
from visualization import visualize
import sys
import os.path
import traceback
import progressbar


def train(args):
  print 'Building computational graph..'
  real_images, real_image_summary = loadData(args)
  G_loss, D_loss, G_summaries, D_summaries = DCGAN(real_images, args)
  D_summaries.append(real_image_summary)
  optimizer_G = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1)
  optimizer_D = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1)
  update_G = slim.learning.create_train_op(G_loss, optimizer_G, 
                                           update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='^generator*'))
  update_D = slim.learning.create_train_op(D_loss, optimizer_D, 
                                           update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='^discriminator*'))
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
            _, summary = sess.run([update_D, D_merged_summary])
            writer.add_summary(summary, global_step)
            # update G
            _, summary = sess.run([update_G, G_merged_summary])
            writer.add_summary(summary, global_step)
            bar.update(epoch_step * args.batch_size)

        checkpoint_filename = os.path.join(args.checkpoint_dir, 'model_%d.ckpt' % epoch)
        saver.save(sess, checkpoint_filename) 
        print '%s saved' % checkpoint_filename

    except Exception as ex:
      traceback.print_exc(file=sys.stdout)
    finally:
      coord.request_stop()

    coord.join(threads)


def sample(args):
  print 'Building computational graph..'
  fake_images = DCGAN(None, args)

  saver = tf.train.Saver()

  print 'Launching session..'
  with tf.Session() as sess:
    print 'Loading model checkpoint %s..' % args.checkpoint
    saver.restore(sess, args.checkpoint)
    
    res = sess.run(fake_images)
    visualize(res, args.output)


if __name__ == '__main__':

  args = parseArgs()
  print args

  if args.mode == 'train':
    train(args)
  else:
    sample(args)

  print 'Exiting..'
