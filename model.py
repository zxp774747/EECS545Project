import tensorflow as tf 
import tensorflow.contrib.slim as slim


def leakyRelu(x, leak=0.2, name='leakyRelu'):
  return tf.maximum(x, leak * x, name=name)


def generator(z, args):
  is_training = (args.mode == 'train')
  with tf.name_scope('generator'):
    with slim.arg_scope([slim.convolution2d_transpose], activation_fn=None, kernel_size=[5, 5], stride=2):
      G = slim.fully_connected(z, 
                              8 * args.g_feature_dim * (args.img_height / 16) * (args.img_width / 16), 
                              activation_fn=tf.nn.relu,
                              scope='generator_full1')
      G = tf.reshape(G, [-1, (args.img_height / 16), (args.img_height / 16), 8 * args.g_feature_dim])
      

      G = slim.convolution2d_transpose(G, 4 * args.g_feature_dim, scope='generator_deconv1')
      G = slim.batch_norm(G, activation_fn=tf.nn.relu, is_training=is_training, scope='generator_batchnorm1')

      G = slim.convolution2d_transpose(G, 2 * args.g_feature_dim, scope='generator_deconv2')
      G = slim.batch_norm(G, activation_fn=tf.nn.relu, is_training=is_training, scope='generator_batchnorm2')

      G = slim.convolution2d_transpose(G, args.g_feature_dim, scope='generator_deconv3')
      G = slim.batch_norm(G, activation_fn=tf.nn.relu, is_training=is_training, scope='generator_batchnorm3')

      G = slim.convolution2d_transpose(G, args.img_depth, scope='generator_deconv4')
      G = slim.batch_norm(G, activation_fn=tf.nn.tanh, is_training=is_training, scope='generator_batchnorm4')

    return G


def discriminator(images, reuse, args):
  is_training = (args.mode == 'train')
  with tf.name_scope('discriminator'):
    with slim.arg_scope([slim.conv2d], activation_fn=None, kernel_size=[5, 5], stride=2, reuse=reuse):
      D = slim.conv2d(images, args.d_feature_dim, scope='discriminator_conv1')
      D = slim.batch_norm(D, activation_fn=leakyRelu, is_training=is_training, reuse=reuse, scope='discriminator_batchnorm1')

      D = slim.conv2d(images, 2 * args.d_feature_dim, scope='discriminator_conv2')
      D = slim.batch_norm(D, activation_fn=leakyRelu, is_training=is_training, reuse=reuse, scope='discriminator_batchnorm2')

      D = slim.conv2d(images, 4 * args.d_feature_dim, scope='discriminator_conv3')
      D = slim.batch_norm(D, activation_fn=leakyRelu, is_training=is_training, reuse=reuse, scope='discriminator_batchnorm3')

      D = slim.conv2d(images, 8 * args.d_feature_dim, scope='discriminator_conv4')
      D = slim.batch_norm(D, activation_fn=leakyRelu, is_training=is_training, reuse=reuse, scope='discriminator_batchnorm4')

      D = slim.flatten(D, scope='discriminator_flatten1')
      D = slim.fully_connected(D, 1, activation_fn=None, reuse=reuse, scope='discriminator_full1')

      return tf.nn.sigmoid(D), D


def DCGAN(real_images, args):
  z = tf.random_uniform([args.batch_size, args.z_dim], minval=-1, maxval=1)
  
  if args.mode == 'train':
    X = tf.sub(tf.mul(real_images, 2.0 / 255), 1)  # scale from [0, 255] to [-1, 1]
    prob_real, logits_real = discriminator(X, False, args)
    fake_images = generator(z, args)
    fake_image_summary = tf.image_summary('fake_images', tf.mul(tf.add(fake_images, 1), 255.0 / 2), max_images=4)
    prob_fake, logits_fake = discriminator(fake_images, True, args)
  
    D_loss_real = slim.losses.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), scope='D_loss_real')
    D_loss_fake = slim.losses.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), scope='D_loss_fake')
    D_loss = tf.add(D_loss_real, D_loss_fake, name='D_loss')
    G_loss = slim.losses.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), scope='G_loss')

    D_loss_real_summary = tf.scalar_summary('D_loss_real', D_loss_real)
    D_loss_fake_summary = tf.scalar_summary('D_loss_fake', D_loss_fake)
    D_loss_summary = tf.scalar_summary('D_loss', D_loss)
    G_loss_summary = tf.scalar_summary('G_loss', G_loss)

    return G_loss, D_loss, [G_loss_summary, fake_image_summary], [D_loss_summary, D_loss_real_summary, D_loss_fake_summary]

  else:  # in sample mode
    fake_images = tf.mul(tf.add(generator(z, args), 1), 255.0 / 2)
    return fake_images
