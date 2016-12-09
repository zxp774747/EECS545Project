from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, sample_size = 64, output_size=96,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, 
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
	#print self.z_dim, self.y_dim
	#print self.image_size
      
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

        self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_size, self.output_size, self.c_dim],
                                        name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.z_sum = tf.histogram_summary("z", self.z)

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits  = self.discriminator(self.images, self.y, reuse=False)
	#print self.G
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
        

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_sum = tf.image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
                                                    
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config, data_X, data_y):
        """Train DCGAN"""
        #print data_y
        #np.random.shuffle(data)
	d_loss_output = []
	g_loss_output = []
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()
	

        self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, 
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        
        sample_images = data_X[0:self.sample_size]
        sample_labels = data_y[0:self.sample_size]
   
            
        counter = 1
        start_time = time.time()

        #if self.load(self.checkpoint_dir):
        #    print(" [*] Load SUCCESS")
        #else:
        #    print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            batch_idxs = min(len(data_X), config.train_size) // config.batch_size
        
            for idx in xrange(0, batch_idxs):
            	batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
            	batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)
		#print self.images.shape
               	#print batch_images.shape
		#print batch_labels.shape
		#print batch_z.shape
            	# Update D network
            	_, summary_str = self.sess.run([d_optim, self.d_sum],
                feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels })
            	self.writer.add_summary(summary_str, counter)

            	# Update G network
            	_, summary_str = self.sess.run([g_optim, self.g_sum],
                feed_dict={ self.z: batch_z, self.y:batch_labels })
            	self.writer.add_summary(summary_str, counter)

           	 # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            	_, summary_str = self.sess.run([g_optim, self.g_sum],
                feed_dict={ self.z: batch_z, self.y:batch_labels })
            	self.writer.add_summary(summary_str, counter)
            
            	errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y:batch_labels})
            	errD_real = self.d_loss_real.eval({self.images: batch_images, self.y:batch_labels})
            	errG = self.g_loss.eval({self.z: batch_z, self.y:batch_labels})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))
		
                if np.mod(counter, 100) == 1:
			samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images, self.y:batch_labels}
                        )
                	save_images(samples, [8, 8],
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    	print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                #if np.mod(counter, 500) == 2:
                #    self.save(config.checkpoint_dir, counter)
	

    def discriminator(self, image, y=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()


	yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
	x = conv_cond_concat(image, yb)
	#print image, yb
	h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
	h0 = conv_cond_concat(h0, yb)

	h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
	h1 = tf.reshape(h1, [self.batch_size, -1])            
	h1 = tf.concat(1, [h1, y])

	h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
	h2 = tf.concat(1, [h2, y])

	h3 = linear(h2, 1, 'd_h3_lin')

	return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        
	s = self.output_size
	s2, s4 = int(s/2), int(s/4) 

	# yb = tf.expand_dims(tf.expand_dims(y, 1),2)
	yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
	z = tf.concat(1, [z, y])

	h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
	h0 = tf.concat(1, [h0, y])

	h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin')))
	h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

	h1 = conv_cond_concat(h1, yb)

	h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2')))
	h2 = conv_cond_concat(h2, yb)

	return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

	s = self.output_size
	s2, s4 = int(s/2), int(s/4)

	# yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
	yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
	z = tf.concat(1, [z, y])

	h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
	h0 = tf.concat(1, [h0, y])

	h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=False))
	h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
	h1 = conv_cond_concat(h1, yb)

	h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=False))
	h2 = conv_cond_concat(h2, yb)

	return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))
