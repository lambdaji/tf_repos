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
tf.app.flags.DEFINE_string("dt_dir", '', "data dt dir")
tf.app.flags.DEFINE_string("model_dir", '/ceph_ai/model/', "model check point dir")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, predict, evaluate, export}")

class DeepFM(object):
    """docstring fo DeepFM."""
    def __init__(self):
        super(DeepFM, self).__init__()
        self.FM_B = None     #bias
        self.FM_W = None     #fm linear weights
        self.FM_V = None     #embedding v

        self.init_graph()

    def init_graph(self):
        """Init a tensorflow Graph containing: input data, variables, model, loss, optimizer"""
        graph = tf.Graph()
        with graph.as_default():
            #------initialize weights------
            self.FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
            self.FM_W = tf.get_variable(name='fm_w', shape=[FLAGS.feature_size], initializer=tf.glorot_normal_initializer())
            self.FM_V = tf.get_variable(name='fm_v', shape=[FLAGS.feature_size,FLAGS.embedding_size], initializer=tf.glorot_normal_initializer())

            #------build feaure-------
            self.iterator = self.input_fn(FLAGS.tr_dir, FLAGS.batch_size, FLAGS.epochs, True)
            self.feat_ids, self.feat_vals, self.labels = self.iterator.get_next()

            #------FM first-order------
            with tf.variable_scope("first-order"):
                feat_wgts = tf.nn.embedding_lookup(self.FM_W, self.feat_ids) # None * F * 1
                y_w = tf.reduce_sum(tf.multiply(feat_wgts, self.feat_vals),1)

            #------FM second-order------
            with tf.variable_scope("second-order"):
                embeddings = tf.nn.embedding_lookup(self.FM_V, self.feat_ids) # None * F * K
                #feat_vals = tf.reshape(self.feat_vals, shape=[-1, FLAGS.field_size, 1])
                embeddings = tf.multiply(embeddings, self.feat_vals) #vij*xi
                sum_square = tf.square(tf.reduce_sum(embeddings,1))
                square_sum = tf.reduce_sum(tf.square(embeddings),1)
                y_v = 0.5*tf.reduce_sum(tf.subtract(sum_square, square_sum),1)	# None * 1

            #------mlp------
            with tf.variable_scope("deep"):
                deep_inputs = tf.reshape(embeddings,shape=[-1,FLAGS.field_size*FLAGS.embedding_size]) # None * (F*K)
                layers = map(int,FLAGS.deep_layers.split(','))
                for i in range(len(layers)):
                    deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
                    weights_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg), scope='mlp%d' % i)

                y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                    weights_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg), scope='deep_out')
                #sig_wgts = tf.get_variable(name='sigmoid_weights', shape=[layers[-1]], initializer=tf.glorot_normal_initializer())
                #sig_bias = tf.get_variable(name='sigmoid_bias', shape=[1], initializer=tf.constant_initializer(0.0))
                #deep_out = tf.nn.xw_plus_b(deep_inputs,sig_wgts,sig_bias,name='deep_out')

            #------sigmoid------
            with tf.variable_scope("deepfm"):
                y_bias = self.FM_B * tf.ones_like(self.labels, dtype=tf.float32)  # None * 1
                self.y = tf.sigmoid(y_bias + y_w + y_v + y_deep)

            #------calc auc------
            self.auc = tf.contrib.metrics.streaming_auc(self.y, self.labels)

            #------bulid loss------
            #lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * 0.001
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y, labels=self.labels)) + \
                FLAGS.l2_reg * tf.nn.l2_loss(self.FM_W) + \
                FLAGS.l2_reg * tf.nn.l2_loss(self.FM_V) #+ \ FLAGS.l2_reg * tf.nn.l2_loss(sig_wgts)

            #------bulid optimizer------
            if FLAGS.optimizer_type == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif FLAGS.optimizer_type == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif FLAGS.optimizer_type == 'Momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.95).minimize(self.loss)
            elif FLAGS.optimizer_type == 'ftrl':
                self.optimizer = tf.train.FtrlOptimizer(FLAGS.learning_rate).minimize(self.loss)

            #------summary for TensorBoard-------
            tf.summary.scalar('tr_loss', self.loss)
            tf.summary.scalar('tr_auc', self.auc)
            #tf.summary.histogram('fm_w_hist', wl)
            #tf.summary.histogram('fm_v_hist', wv)
            #for i in range(len(layers)):
            #	tf.summary.histogram('nn_layer'+str(idx)+'_weights', w_nn_params[idx])
            merged_summary = tf.summary.merge_all()

            #------init--------
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            #return loss, auc, optimizer

    #1 1:0.5 2:0.03519 3:1 4:0.02567 7:0.03708 8:0.01705 9:0.06296 10:0.18185 11:0.02497 12:1 14:0.02565 15:0.03267 17:0.0247 18:0.03158 20:1 22:1 23:0.13169 24:0.02933 27:0.18159 31:0.0177 34:0.02888 38:1 51:1 63:1 132:1 164:1 236:1
    def input_fn(self, ilenames, batch_size=32, num_epochs=1, perform_shuffle=False):
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
        pass

    def train():
        tr_begin = time()
        #self.iterator = input_fn(FLAGS.tr_dir, FLAGS.batch_size, FLAGS.epochs, True)
        while True:
            try:
                self.feat_ids, self.feat_vals, self.labels = self.iterator.get_next()
                #feed_dict = {feat_ids: batch_ids.eval(), feat_vals: batch_vals.eval(), labels: batch_labels}
                loss, auc, _ = sess.run([self.loss, self.auc, self.optimizer])
                print("global_steps=%d	tr_loss=%lf	tr_auc=%lf".format(tf.global_steps, loss, auc))

                if tf.global_steps % FLAGS.eval_steps == 0:
                    va_loss, va_auc = self.evaluate()
                    print("global_steps=%d	va_loss=%lf	va_auc=%lf".format(tf.global_steps, va_loss, va_auc))
            except tf.errors.OutOfRangeError:
                print 'Done training -- epoch limit reached'
                break

        print("Training task time elapsed = %lf".format(time() - tr_begin))

        #Infer task
        self.infer()

    def evaluate(self):
        #tr_begin = time()
        iterator = input_fn(FLAGS.va_dir, FLAGS.batch_size)
        va_loss = []
        va_auc = []
        while True:
            try:
                self.feat_ids, self.feat_vals, self.labels = iterator.get_next()
                #feed_dict = {feat_ids: batch_ids.eval(), feat_vals: batch_vals.eval(), labels: batch_labels}
                loss, auc, _ = sess.run([self.loss, self.auc])
                va_loss.extend(loss)
                va_auc.extend(auc)
            except tf.errors.OutOfRangeError:
                break

        return np.mean(va_loss), np.mean(va_auc)

    def infer(self):
        tr_begin = time()
        iterator = input_fn(FLAGS.va_dir, FLAGS.batch_size)
        with open(FLAGS.pred_file, "w") as fo:
            while True:
                try:
                    self.feat_ids, self.feat_vals, self.labels = iterator.get_next()
                    #feed_dict = {feat_ids: batch_ids.eval(), feat_vals: batch_vals.eval(), labels: batch_labels}
                    y = sess.run([self.y])
                    fo.write("%f\n" % (y))
                except tf.errors.OutOfRangeError:
                    print 'Done infering -- epoch limit reached'
                    break

        print("Infering task time elapsed = %lf".format(time() - tr_begin))

    def export_model():
        # 将训练好的模型保存在当前的文件夹下
        builder = tf.saved_model.builder.SavedModelBuilder(join("./model_name", MODEL_VERSION))
        inputs = {
            "x_wide": tf.saved_model.utils.build_tensor_info(x_wide),
            "x_deep": tf.saved_model.utils.build_tensor_info(x_deep)
            }
        output = {
            "output": tf.saved_model.utils.build_tensor_info(prediction)
            }
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=output,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map = {
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
                }
            )
        builder.save()


    def load_model():
        pass

if __name__ == '__main__':
    #------check Arguments------
    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('tr_dir ', FLAGS.tr_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('epochs ', FLAGS.epochs)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)

    deep_fm = DeepFM()
    iBegin = time()
    if FLAGS.task_type == 'train':
        deep_fm.train()
    elif FLAGS.task_type == 'eval':
        deep_fm.eval()
    elif FLAGS.task_type == 'infer':
        deep_fm.infer()
    print("Time elapsed: %lf".format(time() - iBegin))
