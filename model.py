import tensorflow as tf 
import tensorflow.contrib.slim as slim


def concatFeatureMap(x, y):
  x_shape = x.get_shape()
  y_shape = y.get_shape()
  return tf.concat(3, [x, y * tf.ones([x_shape[0], x_shape[1], x_shape[2], y_shape[3]])])


def leakyRelu(x, leak=0.2, name='leakyRelu'):
  return tf.maximum(x, leak * x, name=name)


def generator(z, y, args):
  with tf.name_scope('generator'):
    with slim.arg_scope([slim.batch_norm], activation_fn=tf.nn.relu, is_training=(args.mode == 'train'), 
                        decay=0.9, epsilon=1e-5, scale=True):
      if args.dataset == 'stl10':
        with slim.arg_scope([slim.convolution2d_transpose], activation_fn=None, kernel_size=[5, 5], stride=2):
          G = slim.fully_connected(z,
                                   8 * 4 * 4 *args.g_feature_dim,
                                   activation_fn=None,
                                   scope='generator_full1')
          G = tf.reshape(G, [-1, 4, 4, 8 * args.g_feature_dim])
          G = slim.batch_norm(G, scope='generator_batchnorm1')
 
          G = slim.convolution2d_transpose(G, 4 * args.g_feature_dim, scope='generator_deconv1')
          G = slim.batch_norm(G, scope='generator_batchnorm2')
        
          G = slim.convolution2d_transpose(G, 2 * args.g_feature_dim, scope='generator_deconv2')
          G = slim.batch_norm(G, scope='generator_batchnorm3')
      
          G = slim.convolution2d_transpose(G, args.g_feature_dim, scope='generator_deconv3')
          G = slim.batch_norm(G, scope='generator_batchnorm4')
      
          G = slim.convolution2d_transpose(G, args.img_depth, activation_fn=tf.nn.tanh, scope='generator_deconv4')
      
      else:
        z = tf.concat(1, [z, y])
        with slim.arg_scope([slim.convolution2d_transpose, slim.fully_connected], activation_fn=None):
          G = slim.fully_connected(z, 1024, scope='generator_full1')
          G = slim.batch_norm(G, scope='generator_batchnorm1')
          G = tf.concat(1, [G, y])

          G = slim.fully_connected(G, 
                                   (args.output_height / 4) * (args.output_width / 4) * 2 * args.g_feature_dim, 
                                   scope='generator_full2')
          G = slim.batch_norm(G, scope='generator_batchnorm2')
          G = tf.reshape(G, [args.batch_size, args.output_height / 4, args.output_height / 4, 2 * args.g_feature_dim])
          G = concatFeatureMap(G, tf.reshape(y, [args.batch_size, 1, 1, -1]))

          G = slim.convolution2d_transpose(G, 2 * args.g_feature_dim, kernel_size=[5, 5], stride=2, scope='generator_deconv1')
          G = slim.batch_norm(G, scope='generator_batchnorm3')
          G = concatFeatureMap(G, tf.reshape(y, [args.batch_size, 1, 1, -1]))
            
          G = slim.convolution2d_transpose(G, args.img_depth, activation_fn=tf.nn.tanh, kernel_size=[5, 5], stride=2, scope='generator_deconv2')

    return G


def discriminator(images, y, reuse, args):
  with tf.name_scope('discriminator'):
    with slim.arg_scope([slim.batch_norm], activation_fn=leakyRelu, is_training=(args.mode == 'train'), 
                        decay=0.9, epsilon=1e-5, scale=True, reuse=reuse):
      with slim.arg_scope([slim.conv2d], activation_fn=None, kernel_size=[5, 5], stride=2, reuse=reuse):
        if args.dataset == 'stl10':
          D = slim.conv2d(images, args.d_feature_dim, scope='discriminator_conv1')
          D = leakyRelu(D)

          D = slim.conv2d(D, 2 * args.d_feature_dim, scope='discriminator_conv2')
          D = slim.batch_norm(D, scope='discriminator_batchnorm1')

          D = slim.conv2d(D, 4 * args.d_feature_dim, scope='discriminator_conv3')
          D = slim.batch_norm(D, scope='discriminator_batchnorm2')

          D = slim.conv2d(D, 8 * args.d_feature_dim, scope='discriminator_conv4')
          D = slim.batch_norm(D, scope='discriminator_batchnorm3')

          D = slim.flatten(D, scope='discriminator_flatten1')
          D = slim.fully_connected(D, 1, activation_fn=None, reuse=reuse, scope='discriminator_full1')
      
        else:
          D = concatFeatureMap(images, tf.reshape(y, [args.batch_size, 1, 1, -1]))
          D = slim.conv2d(D, args.img_depth + 1, activation_fn=leakyRelu, scope='discriminator_conv1')
         
          D = concatFeatureMap(D, tf.reshape(y, [args.batch_size, 1, 1, -1]))
          D = slim.conv2d(D, args.d_feature_dim + args.n_classes, scope='discriminator_conv2')
          D = slim.batch_norm(D, scope='discriminator_batchnorm1')

          D = slim.flatten(D, scope='discriminator_flatten1')
          D = tf.concat(1, [D, y])
          
          D = slim.fully_connected(D, 1024, activation_fn=None, reuse=reuse, scope='discriminator_full1')
          D = slim.batch_norm(D, scope='discriminator_batchnorm2')
          D = tf.concat(1, [D, y])
  
          D = slim.fully_connected(D, 1, activation_fn=None, reuse=reuse, scope='discriminator_full2')

      return tf.nn.sigmoid(D), D


def DCGAN(real_images, z, y, args):
  scaled_fake_images = generator(z, y, args)
  fake_images = tf.mul(tf.div(tf.add(scaled_fake_images, 1), 2), 255)
  
  if args.mode == 'train':
    scaled_real_images = tf.sub(tf.div(tf.mul(real_images, 2), 255), 1)  # scale from [0, 255] to [-1, 1]
    scaled_real_image_summary = tf.histogram_summary('scaled_real_image_summary', scaled_real_images)
    prob_real, logits_real = discriminator(scaled_real_images, y, False, args)
    prob_real_summary = tf.histogram_summary('prob_real', prob_real)
    scaled_fake_image_summary = tf.histogram_summary('scaled_fake_image_summary', scaled_fake_images)
    prob_fake, logits_fake = discriminator(scaled_fake_images, y, True, args)
    prob_fake_summary = tf.histogram_summary('prob_fake', prob_fake) 
 
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_real, tf.ones_like(logits_real), name='D_loss_real'))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.zeros_like(logits_fake), name='D_loss_fake'))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.ones_like(logits_fake), name='G_loss'))

    D_loss_real_summary = tf.scalar_summary('D_loss_real', D_loss_real)
    D_loss_fake_summary = tf.scalar_summary('D_loss_fake', D_loss_fake)
    D_loss_summary = tf.scalar_summary('D_loss', D_loss)
    G_loss_summary = tf.scalar_summary('G_loss', G_loss)


    return (G_loss, D_loss, fake_images,
           [G_loss_summary], 
           [D_loss_summary, D_loss_real_summary, D_loss_fake_summary, prob_real_summary, prob_fake_summary, scaled_real_image_summary, scaled_fake_image_summary])

  else:  # in sample mode
    return fake_images
