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

class DeepFM(object):
    """docstring fo DeepFM."""
    def __init__(self, arg):
        super(DeepFM, self).__init__()
        self.fw_b = None     #bias
        self.fw_l = None     #fm linear weights
        self.fw_v = None     #embedding v

    def init_graph(self):
        """Init a tensorflow Graph containing: input data, variables, model, loss, optimizer"""
        graph = tf.Graph()
        with graph.as_default():
            #------model weights------
	        self.fw_b = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
	        self.fw_l = tf.get_variable(name='fm_w', shape=[FLAGS.feature_size], initializer=tf.glorot_normal_initializer())
	        self.fw_v = tf.get_variable(name='fm_v', shape=[FLAGS.feature_size,FLAGS.embedding_size], initializer=tf.glorot_normal_initializer())

            y = model_fn()

    #1 1:0.5 2:0.03519 3:1 4:0.02567 7:0.03708 8:0.01705 9:0.06296 10:0.18185 11:0.02497 12:1 14:0.02565 15:0.03267 17:0.0247 18:0.03158 20:1 22:1 23:0.13169 24:0.02933 27:0.18159 31:0.0177 34:0.02888 38:1 51:1 63:1 132:1 164:1 236:1
    def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
        print('Parsing', filenames)
        def decode_libsvm(line):
            #columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
            #features = dict(zip(CSV_COLUMNS, columns))
            #labels = features.pop(LABEL_COLUMN)
            columns = tf.string_split([line], ' ')
            labels = tf.string_to_number(columns.values[0], out_type=tf.int32)
            splits = tf.string_split(columns.values[1:], ':')
            id_vals = tf.reshape(splits.values,splits.dense_shape)
            feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
            feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
            feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
            #for i in range(splits.dense_shape.eval()[0]):
            #    feat_ids.append(tf.string_to_number(splits.values[2*i], out_type=tf.int32))
            #    feat_vals.append(tf.string_to_number(splits.values[2*i+1]))
            #return tf.reshape(feat_ids,shape=[-1,FLAGS.field_size]), tf.reshape(feat_vals,shape=[-1,FLAGS.field_size]), labels
            return feat_ids, feat_vals, labels

        # Extract lines from input files using the Dataset API, can pass one filename or filename list
        dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)    # multi-thread pre-process then prefetch

        # Randomizes input using a window of 256 elements (read into memory)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)

        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size) # Batch size to use

        return dataset.make_one_shot_iterator()
        #iterator = dataset.make_one_shot_iterator()
        #batch_ids, batch_vals, batch_labels = iterator.get_next()
        #return tf.reshape(batch_ids,shape=[-1,FLAGS.field_size]), tf.reshape(batch_vals,shape=[-1,FLAGS.field_size]), batch_labels
        #return tf.reshape(batch_ids,shape=[-1,FLAGS.field_size]), batch_vals, batch_labels


    def model_fn(self, feat_ids, feat_vals, labels):
        """bulid f(x)"""
        #------FM first-order------
        with tf.variable_scope("first-order"):
        	feat_wgts = tf.nn.embedding_lookup(wl, feat_ids) # None * F * 1
        	fm_w = tf.reduce_sum(tf.mulitply(feat_wgts,feat_vals),1)

        #------FM second-order------
        with tf.variable_scope("second-order"):
        	embeddings = tf.nn.embedding_lookup(wv, feat_ids) # None * F * K
            feat_vals = tf.reshape(feat_vals, shape=[-1, FLAGS.field_size, 1])
        	embeddings = tf.multiply(embeddings,feat_vals) #vij*xi
        	sum_square = tf.square(tf.reduce_sum(embeddings,1))
        	square_sum = tf.reduce_sum(tf.square(embeddings),1)
        	fm_v = 0.5*tf.reduce_sum(tf.subtract(sum_square, square_sum),1)	# None * 1

        #------mlp------
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

        #------sigmoid------
        with tf.variable_scope("deepfm"):
        	bias = wb * tf.ones_like(self.train_labels)  # None * 1
        	y = tf.sigmoid(bias + fm_w + fm_v + deep_out)

        #------calc auc------
    	auc = tf.contrib.metrics.streaming_auc(y,label)

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

        #return loss, auc, optimizer

    def train():
        tr_begin = time()
	    iterator = input_fn(FLAGS.tr_dir, True, FLAGS.batch_size, FLAGS.epochs)
  	    while True:
    	try:
			batch_ids, batch_vals, batch_labels = iterator.get_next()
            loss = model_fn(batch_ids, batch_vals, batch_labels)
            #feature to map to id
			feed_dict = {feat_ids: batch_ids.eval(), feat_vals: batch_vals.eval(), labels: batch_labels}
        	loss, auc = sess.run((loss, auc), feed_dict=feed_dict)
      		print("global_steps=%d	tr_loss=%lf	tr_auc=%lf".format(tf.global_steps, loss, auc))

			if tf.global_steps % 10000 == 0:
				va_loss, va_auc = evaluate()
				print("global_steps=%d	va_loss=%lf	va_auc=%lf".format(tf.global_steps, va_loss, va_auc))
    	except tf.errors.OutOfRangeError:
      		break

    def evaluate(self):
        pass

    def infer(self):
        pass


'''
Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
'''
graph = tf.Graph()
sess = tf.Session()
with graph.as_default():
	#------bulid feature------

	feat_ids  = tf.placeholder(tf.int32,shape=[None,None],name='feat_ids') #batch_size*F
	feat_vals = tf.placeholder(tf.float32,shape=[None,None],name='feat_vals')
	label = tf.placeholder(tf.float32, shape=[None,1], name='label')  # None * 1

	#------model weights------
	wb = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
	wl = tf.get_variable(name='fm_w', shape=[FLAGS.feature_size], initializer=tf.glorot_normal_initializer())
	wv = tf.get_variable(name='fm_v', shape=[FLAGS.feature_size,FLAGS.embedding_size], initializer=tf.glorot_normal_initializer())

	#------bulid f(x)------
	with tf.variable_scope("first-order"):
		feat_wgts = tf.nn.embedding_lookup(wl, feat_ids) # None * F * 1
		fm_w = tf.reduce_sum(tf.mulitply(feat_wgts,feat_vals),1)

	with tf.variable_scope("second-order"):
		embeddings = tf.nn.embedding_lookup(wv, feat_ids) # None * F * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, FLAGS.field_size, 1])
		embeddings = tf.multiply(embeddings,feat_vals) #vij*xi
		sum_square = tf.square(tf.reduce_sum(embeddings,1))
		square_sum = tf.reduce_sum(tf.square(embeddings),1)
		fm_v = 0.5*tf.reduce_sum(tf.subtract(sum_square, square_sum),1)	# None * 1

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



	#------summary for TensorBoard-------
    tf.summary.scalar('tr_loss', loss)
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

#Constants
xgb_trees = 100
# Column Title
CSV_COLUMNS = [ "is_click","u_pl","u_ppvn","u_de","u_os","u_t","a_m_w","a_b_w","c_h","c_w","c_al",
                "u_ctr","a_a_ctr","a_t_ctr","c_q_ctr","c_al_ctr","c_n_ctr","c_t_ctr","c_t_n_ctr",
                "u_a_city_ctr","u_a_age_ctr","u_a_x_ctr","u_a_g_ctr","u_a_c_ctr","c_q_a_ctr","c_q_t_sim","c_q_adtype_ctr","c_mw_a_ctr" ]
XGB_COLUMNS = [ 'xgbf_%d' % i for i in range(xgb_trees) ]
CSV_COLUMNS = CSV_COLUMNS + XGB_COLUMNS
LABEL_COLUMN = "is_click"

# Columns Defaults
CSV_COLUMN_DEFAULTS = [[0],["ADR"],["7.9"],["PHONE"],["5.1"],["0"], [""],[""],["17"],[""],["0"]]
CSV_COLUMN_CTR_DEFAULTS = [[0.0] for i in range(17)]
XGB_COLUMN_DEFAULTS = [[0] for i in range(xgb_trees)]
CSV_COLUMN_DEFAULTS = CSV_COLUMN_DEFAULTS + CSV_COLUMN_CTR_DEFAULTS + XGB_COLUMN_DEFAULTS







def evaluate():
	pass

def infer():
	pass

def
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
