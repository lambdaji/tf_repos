#!/usr/bin/env python
#coding=utf-8
"""
TensorFlow Implementation of <<Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate>>
and <<Product-based Neural Networks for User Response Prediction>> with the fellowing features：
#1 Input pipline using Dataset high level API, Support parallel and prefetch reading
#2 Train pipline using Coustom Estimator by rewriting model_fn
#3 Support distincted training by TF_CONFIG
#4 Support export servable model for TensorFlow Serving

https://zhuanlan.zhihu.com/p/37562283
https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408

by lambdaji
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import argparse
import shutil
#import sys
import os
import json
import glob
from datetime import date, timedelta
from time import time

import random
import tensorflow as tf

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.app.flags.DEFINE_integer("feature_size", 0, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 0, "Number of common fields")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_float("ctr_task_wgt", 0.5, "loss weight of ctr task")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

#40362692,0,0,216:9342395:1.0 301:9351665:1.0 205:7702673:1.0 206:8317829:1.0 207:8967741:1.0 508:9356012:2.30259 210:9059239:1.0 210:9042796:1.0 210:9076972:1.0 210:9103884:1.0 210:9063064:1.0 127_14:3529789:2.3979 127_14:3806412:2.70805
def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    def _parse_fn(record):
        features = {
            "y": tf.FixedLenFeature([], tf.float32),
            "z": tf.FixedLenFeature([], tf.float32),
            "feat_ids": tf.FixedLenFeature([FLAGS.field_size], tf.int64),
            #"feat_vals": tf.FixedLenFeature([None], tf.float32),
            "u_catids": tf.VarLenFeature(tf.int64),
            "u_catvals": tf.VarLenFeature(tf.float32),
            "u_shopids": tf.VarLenFeature(tf.int64),
            "u_shopvals": tf.VarLenFeature(tf.float32),
            "u_intids": tf.VarLenFeature(tf.int64),
            "u_intvals": tf.VarLenFeature(tf.float32),
            "u_brandids": tf.VarLenFeature(tf.int64),
            "u_brandvals": tf.VarLenFeature(tf.float32),
            "a_catids": tf.FixedLenFeature([], tf.int64),
            "a_shopids": tf.FixedLenFeature([], tf.int64),
            "a_brandids": tf.FixedLenFeature([], tf.int64),
            "a_intids": tf.VarLenFeature(tf.int64)
        }
        parsed = tf.parse_single_example(record, features)
        y = parsed.pop('y')
        z = parsed.pop('z')
        return parsed, {"y": y, "z": z}

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TFRecordDataset(filenames).map(_parse_fn, num_parallel_calls=10).prefetch(500000)    # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use
    #dataset = dataset.padded_batch(batch_size, padded_shapes=({"feeds_ids": [None], "feeds_vals": [None], "title_ids": [None]}, [None]))   #不定长补齐

    #return dataset.make_one_shot_iterator()
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    #return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels

def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    #------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    #optimizer = params["optimizer"]
    layers = map(int, params["deep_layers"].split(','))
    dropout = map(float, params["dropout"].split(','))
    ctr_task_wgt = params["ctr_task_wgt"]
    common_dims = field_size*embedding_size

    #------bulid weights------
    Feat_Emb = tf.get_variable(name='embeddings', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feaure-------
    #{U-A-X-C不需要特殊处理的特征}
    feat_ids    = features['feat_ids']
    #feat_vals   = features['feat_vals']
    #{User multi-hot}
    u_catids    = features['u_catids']
    u_catvals   = features['u_catvals']
    u_shopids   = features['u_shopids']
    u_shopvals  = features['u_shopvals']
    u_intids    = features['u_intids']
    u_intvals   = features['u_intvals']
    u_brandids  = features['u_brandids']
    u_brandvals = features['u_brandvals']
    #{Ad}
    a_catids    = features['a_catids']
    a_shopids   = features['a_shopids']
    a_brandids  = features['a_brandids']
    a_intids    = features['a_intids']      #multi-hot
    #{X multi-hot}
    #x_intids    = features['x_intids']
    #x_intvals   = features['x_intvals']

    y = labels['y']
    z = labels['z']
    #y = tf.Print(y, [y], message="This is y: ")
    #z = tf.Print(z, [z], message="This is z: ")

    #------build f(x)------
    with tf.variable_scope("Shared-Embedding-layer"):
        common_embs = tf.nn.embedding_lookup(Feat_Emb, feat_ids)     # None * F' * K
        #common_embs = tf.multiply(common_embs, feat_vals)
        u_cat_emb   = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_catids,  sp_weights=u_catvals,   combiner="sum")               # None * K
        u_shop_emb  = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_shopids, sp_weights=u_shopvals,  combiner="sum")
        u_brand_emb = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_brandids,sp_weights=u_brandvals, combiner="sum")
        u_int_emb   = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_intids,  sp_weights=u_intvals,   combiner="sum")
        a_int_emb   = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=a_intids,  sp_weights=None,        combiner="sum")
        a_cat_emb   = tf.nn.embedding_lookup(Feat_Emb, a_catids)
        a_shop_emb  = tf.nn.embedding_lookup(Feat_Emb, a_shopids)
        a_brand_emb = tf.nn.embedding_lookup(Feat_Emb, a_brandids)

        x_concat = tf.concat([tf.reshape(common_embs,shape=[-1, common_dims]),u_cat_emb,u_shop_emb,u_brand_emb,u_int_emb,a_cat_emb,a_shop_emb,a_brand_emb,a_int_emb],axis=1)    # None * (F * K)

    with tf.name_scope("CVR_Task"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False
        x_cvr = x_concat
        for i in range(len(layers)):
            x_cvr = tf.contrib.layers.fully_connected(inputs=x_cvr, num_outputs=layers[i], \
            	weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='cvr_mlp%d' % i)

            if FLAGS.batch_norm:
				x_cvr = batch_norm_layer(x_cvr, train_phase=train_phase, scope_bn='cvr_bn_%d' %i)   	#放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
            if mode == tf.estimator.ModeKeys.TRAIN:
				x_cvr = tf.nn.dropout(x_cvr, keep_prob=dropout[i])                              	#Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)

        y_cvr = tf.contrib.layers.fully_connected(inputs=x_cvr, num_outputs=1, activation_fn=tf.identity, \
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='cvr_out')
        y_cvr = tf.reshape(y_cvr,shape=[-1])

    with tf.name_scope("CTR_Task"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False

        x_ctr = x_concat
        for i in range(len(layers)):
            x_ctr = tf.contrib.layers.fully_connected(inputs=x_ctr, num_outputs=layers[i], \
            	weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='ctr_mlp%d' % i)

            if FLAGS.batch_norm:
				x_ctr = batch_norm_layer(x_ctr, train_phase=train_phase, scope_bn='ctr_bn_%d' %i)   	#放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
            if mode == tf.estimator.ModeKeys.TRAIN:
				x_ctr = tf.nn.dropout(x_ctr, keep_prob=dropout[i])                              	#Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)

        y_ctr = tf.contrib.layers.fully_connected(inputs=x_ctr, num_outputs=1, activation_fn=tf.identity, \
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='ctr_out')
        y_ctr = tf.reshape(y_ctr,shape=[-1])

    with tf.variable_scope("MTL-Layer"):
        pctr = tf.sigmoid(y_ctr)
        pcvr = tf.sigmoid(y_cvr)
        pctcvr = pctr*pcvr

    predictions={"pcvr": pcvr, "pctr": pctr, "pctcvr": pctcvr}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------bulid loss------
    ctr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_ctr, labels=y))
    #cvr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_ctcvr, labels=z))
    cvr_loss = tf.reduce_mean(tf.losses.log_loss(predictions=pctcvr, labels=z))
    loss = ctr_task_wgt * ctr_loss + (1  -ctr_task_wgt) * cvr_loss + l2_reg * tf.nn.l2_loss(Feat_Emb)

    tf.summary.scalar('ctr_loss', ctr_loss)
    tf.summary.scalar('cvr_loss', cvr_loss)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "CTR_AUC": tf.metrics.auc(y, pctr),
        "CVR_AUC": tf.metrics.auc(z, pcvr),
        "CTCVR_AUC": tf.metrics.auc(z, pctcvr)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    #------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z

def set_dist_env():
    if FLAGS.dist_mode == 1:        # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        ps_hosts = FLAGS.ps_hosts.split(',')
        chief_hosts = FLAGS.chief_hosts.split(',')
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # 无worker参数
        tf_config = {
            'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif FLAGS.dist_mode == 2:      # 集群分布式模式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1] # get first worker as chief
        worker_hosts = worker_hosts[2:] # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
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

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(_):
    #------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir
    #FLAGS.data_dir  = FLAGS.data_dir + FLAGS.dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)
    print('ctr_task_wgt ', FLAGS.ctr_task_wgt)

    #------init Envs------
    tr_files = glob.glob("%s/tr/*tfrecord" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/te/*tfrecord" % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%s/te/*tfrecord" % FLAGS.data_dir)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    set_dist_env()

    #------bulid Tasks------
    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
		"dropout": FLAGS.dropout,
        "ctr_task_wgt":FLAGS.ctr_task_wgt
    }
    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
            log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    Estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(Estimator, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        Estimator.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'infer':
        preds = Estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        with open(FLAGS.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\t%f\n" % (prob['pctr'], prob['pcvr']))
    elif FLAGS.task_type == 'export':
        print("Not Implemented, Do It Yourself!")
        #feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)

        #feature_spec = {
        #    'feat_ids': tf.FixedLenFeature(dtype=tf.int64, shape=[None, FLAGS.field_size]),
        #    'feat_vals': tf.FixedLenFeature(dtype=tf.float32, shape=[None, FLAGS.field_size])
        #}
        #serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

        #feature_spec = {
        #    'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
        #    'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        #}
        #serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        #Estimator.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
