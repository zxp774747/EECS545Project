import tensorflow as tf
import tensorflow.contrib.slim as slim
from data import loadData
from model import DCGAN
from args import parseArgs
import sys


def train(args):
  print 'Building computational graph..'
  real_images, real_image_summary = loadData(args)
  G_loss, D_loss, G_loss_summary, D_loss_summary, fake_image_summary = DCGAN(real_images, args)
  optimizer_G = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1)
  optimizer_D = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1)
  update_G = slim.learning.create_train_op(G_loss, optimizer_G, 
                                           update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='^generator*'), summarize_gradients=True)
  update_D = slim.learning.create_train_op(D_loss, optimizer_D, 
                                           update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='^discriminator*'), summarize_gradients=True)
  G_merged_summary = tf.merge_summary([fake_image_summary, G_loss_summary])
  D_merged_summary = tf.merge_summary([real_image_summary, D_loss_summary])

  print 'Launching session..'
  with tf.Session() as sess:
    print 'Initializing variables..'
    sess.run(tf.initialize_all_variables())

    print 'Staring the queue..'
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print 'Training..'
    writer = tf.train.SummaryWriter(args.log_dir)
    try:
      step = 0
      while not coord.should_stop():
        step += 1
        # update D
        _, summary = sess.run([update_D, D_merged_summary])
        writer.add_summary(summary, step)
        # update G twice
        _, summary = sess.run([update_G, G_merged_summary])
        writer.add_summary(summary, step)
        _, summary = sess.run([update_G, G_merged_summary])
        writer.add_summary(summary, step)

    except Exception as ex:
      traceback.print_exc(file=sys.stdout)
    finally:
      coord.request_stop()

    coord.join(threads)


def sample(args):
  pass


if __name__ == '__main__':

  args = parseArgs()
  print args

  if args.mode == 'train':
    train(args)
  else:
    sample(args)

  print 'Exiting..'