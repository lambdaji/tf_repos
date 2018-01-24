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
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)    # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    #return dataset.make_one_shot_iterator()
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    #return tf.reshape(batch_ids,shape=[-1,FLAGS.field_size]), tf.reshape(batch_vals,shape=[-1,FLAGS.field_size]), batch_labels
    return batch_features, batch_labels

def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    #------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    feature_size = params["feature_size"]
    layers = map(int, params["deep_layers"].split(','))

    #------bulid weights------
    FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
    FM_W = tf.get_variable(name='fm_w', shape=[FLAGS.feature_size], initializer=tf.glorot_normal_initializer())
    FM_V = tf.get_variable(name='fm_v', shape=[FLAGS.feature_size,FLAGS.embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feaure-------
    feat_ids  = features['feat_ids']
    feat_vals = features['feat_vals']

    #------FM first-order------
    with tf.variable_scope("first-order"):
        feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids) # None * F * 1
        y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals),1)

    #------FM second-order------
    with tf.variable_scope("second-order"):
        embeddings = tf.nn.embedding_lookup(FM_V, feat_ids) # None * F * K
        #feat_vals = tf.reshape(feat_vals, shape=[-1, FLAGS.field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals) #vij*xi
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
        y_bias = FM_B * tf.ones_like(labels, dtype=tf.float32)  # None * 1
        y = tf.sigmoid(y_bias + y_w + y_v + y_deep)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"prob": y})

    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y))
    eval_metric_ops = {"auc": tf.metrics.auc(labels, pred) }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"prob": y},
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    optimizer = tf.train.GradientDescentOptimizer(lr)
    if FLAGS.optimizer == "adagrad":
        optimizer = tf.train.AdagradOptimizer(lr)
    elif FLAGS.optimizer == "adadelta":
        optimizer = tf.train.AdadeltaOptimizer(lr)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, pred, loss, train_op)

    #------calc auc------
    auc = tf.contrib.metrics.streaming_auc(y, labels)

    #------bulid loss------
    #lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * 0.001
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
        FLAGS.l2_reg * tf.nn.l2_loss(FM_W) + \
        FLAGS.l2_reg * tf.nn.l2_loss(FM_V) #+ \ FLAGS.l2_reg * tf.nn.l2_loss(sig_wgts)

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
    tf.summary.scalar('tr_auc', auc)
    #tf.summary.histogram('fm_w_hist', wl)
    #tf.summary.histogram('fm_v_hist', wv)
    #for i in range(len(layers)):
    #	tf.summary.histogram('nn_layer'+str(idx)+'_weights', w_nn_params[idx])
    merged_summary = tf.summary.merge_all()

    #------init--------
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #return loss, auc, optimizer

def train():
    tr_begin = time()
    #iterator = input_fn(FLAGS.tr_dir, FLAGS.batch_size, FLAGS.epochs, True)
    while True:
        try:
            feat_ids, feat_vals, labels = iterator.get_next()
            #feed_dict = {feat_ids: batch_ids.eval(), feat_vals: batch_vals.eval(), labels: batch_labels}
            loss, auc, _ = sess.run([loss, auc, optimizer])
            print("global_steps=%d	tr_loss=%lf	tr_auc=%lf".format(tf.global_steps, loss, auc))

            if tf.global_steps % FLAGS.eval_steps == 0:
                va_loss, va_auc = evaluate()
                print("global_steps=%d	va_loss=%lf	va_auc=%lf".format(tf.global_steps, va_loss, va_auc))
        except tf.errors.OutOfRangeError:
            print 'Done training -- epoch limit reached'
            break

    print("Training task time elapsed = %lf".format(time() - tr_begin))

    #Infer task
    infer()

def evaluate(self):
    #tr_begin = time()
    iterator = input_fn(FLAGS.va_dir, FLAGS.batch_size)
    va_loss = []
    va_auc = []
    while True:
        try:
            feat_ids, feat_vals, labels = iterator.get_next()
            #feed_dict = {feat_ids: batch_ids.eval(), feat_vals: batch_vals.eval(), labels: batch_labels}
            loss, auc, _ = sess.run([loss, auc])
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
                feat_ids, feat_vals, labels = iterator.get_next()
                #feed_dict = {feat_ids: batch_ids.eval(), feat_vals: batch_vals.eval(), labels: batch_labels}
                y = sess.run([y])
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

def main(_):
    if not FLAGS.local_mode:
        # 数据工厂分布式模式
        if FLAGS.dist_mode:
             ps_hosts = FLAGS.ps_hosts.split(',')
             worker_hosts = FLAGS.worker_hosts.split(',')
             chief_hosts = worker_hosts[0:1] # get first worker as chief
             worker_hosts = worker_hosts[2:] # the rest as worker
             task_index = FLAGS.task_index
             job_name = FLAGS.job_name
             print('ps_host', FLAGS.ps_hosts)
             print('worker_host', FLAGS.worker_hosts)
             print('chief_hosts', FLAGS.chief_hosts)
             print('job_name', FLAGS.job_name)
             print('task_index', str(FLAGS.task_index))
             # use #worker=0 as chief
             if job_name == "worker" and task_index == 0:
                 job_name = "chief"
             # use #worker=1 as evaluator
             if job_name == "worker" and task_index == 1:
                 job_name = 'evaluator'
                 task_index = 0
             # the others as worker
             if job_name == "worker" and task_index > 1:
                 task_index -= 2

             tf_config = {'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
                          'task': {'type': job_name, 'index': task_index }}
             print(json.dumps(tf_config))
             os.environ['TF_CONFIG'] = json.dumps(tf_config)
        # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        else:
             ps_hosts = FLAGS.ps_hosts.split(',')
             chief_hosts = FLAGS.chief_hosts.split(',')
             task_index = FLAGS.task_index
             job_name = FLAGS.job_name
             print('ps_host', FLAGS.ps_hosts)
             print('chief_hosts', FLAGS.chief_hosts)
             print('job_name', FLAGS.job_name)
             print('task_index', str(FLAGS.task_index))
             # 无worker参数
             tf_config = {'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
                          'task': {'type': job_name, 'index': task_index }}
             print(json.dumps(tf_config))
             os.environ['TF_CONFIG'] = json.dumps(tf_config)

    train_files = glob.glob("%s/simnet.000*.tfrecords" % FLAGS.data_dir)
    random.shuffle(train_files)
    print("train_files:", train_files)
    eval_files = glob.glob("%s/simnet.001*.tfrecords" % FLAGS.data_dir)
    print("eval_flies:", eval_files)

    print("learning_rate=", FLAGS.lr)
    model_params = {"learning_rate": FLAGS.lr,
                    "emb_size": FLAGS.emb_size,
                    "vocab_size": FLAGS.vocab_size,
                    "fc_size": FLAGS.fc_size,
                    "max_seq_len": FLAGS.max_seq_len,
                    "win_size": FLAGS.win_size,
                    "kernel_size": FLAGS.kernel_size}

    config = tf.estimator.RunConfig().replace(
            session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
            log_step_count_steps=10000,
            save_summary_steps=10000)

    simnet_bow = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=FLAGS.model_dir,
            params=model_params,
            config=config)
    train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(train_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
    eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(eval_files, num_epochs=1, batch_size=FLAGS.batch_size),
            steps=None,
            start_delay_secs=1000,
            throttle_secs=1200)
    tf.estimator.train_and_evaluate(simnet_bow, train_spec, eval_spec)

    #predictions = simnet_bow.predict(input_fn=lambda: input_fn(eval_files, num_epochs=1))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run() 

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
