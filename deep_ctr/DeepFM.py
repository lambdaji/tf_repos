#!/usr/bin/env python
#coding=utf-8

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import argparse
#import shutil
#import sys
#import os
#import glob
#from datetime import date, timedelta
from time import time
#import gc
#from multiprocessing import Process

#import math
import pandas as pd
import numpy as np
import tensorflow as tf

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("feature_size", 16, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 16, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
tf.app.flags.DEFINE_integer("epochs", 16, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 16, "Number of batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.01, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '128,64,32', "deep layers")
tf.app.flags.DEFINE_string("tr_dir", '/ceph_ai/data/', "train data dir")
tf.app.flags.DEFINE_string("model_dir", '/ceph_ai/model/', "model check point dir")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, predict, evaluate}")

'''
Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
'''
graph = tf.Graph()
sess = tf.Session()
with graph.as_default():
	#------bulid feature------
	feat_ids  = tf.placeholder(tf.int32,shape=[None,None],name='feat_ids') #batch_size*F
	feat_vals = tf.placeholder(tf.float,shape=[None,None],name='feat_vals')
	label = tf.placeholder(tf.float32, shape=[None,1], name="label")  # None * 1

	#------model weights------
	wb = tf.get_variable(name="fm_bias", shape=[1], initializer=tf.constant_initializer(0.0))
	wl = tf.get_variable(name='fm_w', shape=[FLAGS.feature_size], initializer=tf.glorot_normal_initializer())
	wv = tf.get_variable(name='fm_v', shape=[FLAGS.feature_size,FLAGS.embedding_size], initializer=tf.glorot_normal_initializer())

	#------bulid f(x)------
	with tf.variable_scope("first-order"):
		feat_wgts = tf.nn.embedding_lookup(wl, feat_ids) # None * F * 1
		fm_w = tf.reduce_sum(tf.mulitply(feat_wgts,feat_vals),1)

	with tf.variable_scope("second-order"):
		embeddings = tf.nn.embedding_lookup(wv, feat_ids) # None * F * 1
		embeddings = tf.mulitply(embeddings,feat_vals) #vij*xi
		sum_square = tf.square(tf.reduce_sum(embeddings,1))
		square_sum = tf.reduce_sum(tf.square(embeddings),1)
		fm_v = 0.5*tf.reduce_sum(tf.subtract(sum_square - square_sum),1)	# None * 1

	with tf.variable_scope("deep"):
		deep_inputs = tf.reshape(embeddings,shape=[-1,FLAGS.field_size*FLAGS.embedding_size]) # None * (F*K)
		layers = map(int,FLAGS.deep_layers.split(','))
		for i in range(len(layers)):
			deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
							weights_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg), scope='mlp%d' % i)
		deep_out = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
							weights_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg), scope='deep_out')
		#sig_wgts = tf.get_variable(name='sigmoid_weights', shape=[layers[-1]], initializer=tf.glorot_normal_initializer())
		#sig_bias = tf.get_variable(name='sigmoid_bias', shape=[1], initializer=tf.constant_initializer(0.0))
		#deep_out = tf.nn.xw_plus_b(deep_inputs,sig_wgts,sig_bias,name='deep_out')

	with tf.variable_scope("deepfm"):
		bias = wb * tf.ones_like(self.train_labels)  # None * 1
		y = tf.sigmoid(bias + fm_w + fm_v + deep_out)

	#------bulid loss------
	#lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * 0.001
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=label)) + \
                        FLAGS.l2_reg * tf.nn.l2_loss(wl) + \
                        FLAGS.l2_reg * tf.nn.l2_loss(wv) #+ \ FLAGS.l2_reg * tf.nn.l2_loss(sig_wgts)

	#------bulid optimizer------
	if FLAGS.optimizer_type == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)
    elif FLAGS.optimizer_type == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate, initial_accumulator_value=1e-8).minimize(loss)
    elif FLAGS.optimizer_type == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.95).minimize(loss)
    elif FLAGS.optimizer_type == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(FLAGS.learning_rate).minimize(loss)

	#------summary for TensorBoard-------
    tf.summary.scalar('tr_loss', loss)
	auc = tf.contrib.metrics.streaming_auc(y,label)
	tf.summary.scalar('tr_auc', auc)
    #tf.summary.histogram('fm_w_hist', wl)
    #tf.summary.histogram('fm_v_hist', wv)
    #for i in range(len(layers)):
	#	tf.summary.histogram('nn_layer'+str(idx)+'_weights', w_nn_params[idx])
    merged_summary = tf.summary.merge_all()

	#------init--------
	init = tf.global_variables_initializer()
    #sess = tf.Session()
    sess.run(init)

def input_fn(file_path, perform_shuffle=False, batch_size=32, repeat_count=1):
   def decode_csv(line):
       parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
       label = parsed_line[-1:] # Last element is the label
       del parsed_line[-1] # Delete last element
       features = parsed_line # Everything (but last element) are the features
       d = dict(zip(feature_names, features)), label
       return d

   dataset = (tf.data.TextLineDataset(file_path) # Read text file
       .skip(1) # Skip header row
       .map(decode_csv)) # Transform each elem by applying decode_csv fn
   if perform_shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
       dataset = dataset.shuffle(buffer_size=256)
   dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   dataset = dataset.batch(batch_size)  # Batch size to use
   return dataset.make_one_shot_iterator()
   #batch_features, batch_labels = iterator.get_next()
   #return batch_features, batch_labels


def train():
	tr_begin = time()
	iterator = input_fn(FLAGS.tr_dir, True, FLAGS.batch_size, FLAGS.epochs)
  	while True:
    	try:
			batch_features, batch_labels = iterator.get_next()
			feed_dict = {feat_ids: batch_features.keys(), feat_vals: batch_features.values(), labels: batch_labels}
        	loss, auc = sess.run((loss, auc), feed_dict=feed_dict)
      		print("global_steps=%d	tr_loss=%lf	tr_auc=%lf".format(tf.global_steps, loss, auc))

			if tf.global_steps % 10000 == 0:
				va_loss, va_auc = evaluate()
				print("global_steps=%d	va_loss=%lf	va_auc=%lf".format(tf.global_steps, va_loss, va_auc))
    	except tf.errors.OutOfRangeError:
      		break

def evaluate():
	pass

def predict():
	pass


if __name__ == '__main__':
    # Data loading

	iBegin=time()
	if FLAGS.task_type == 'train':
		train()
	elif FLAGS.task_type == 'train':
		evaluate()
	elif FLAGS.task_type == 'train':
		predict()
	print("time elapsed: ", time()-iBegin)
