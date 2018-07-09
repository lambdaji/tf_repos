#!/usr/bin/env python
#coding=utf-8
"""
TensorFlow Implementation of <<Deep Interest Network for Click-Through Rate Prediction>>
Dataset desc: https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408

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
tf.app.flags.DEFINE_integer("field_size", 0, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("attention_pooling", True, "attention pooling")
tf.app.flags.DEFINE_string("attention_layers", '256', "Attention Net mlp layers")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

#1 1:0.5 2:0.03519 3:1 4:0.02567 7:0.03708 8:0.01705 9:0.06296 10:0.18185 11:0.02497 12:1 14:0.02565 15:0.03267 17:0.0247 18:0.03158 20:1 22:1 23:0.13169 24:0.02933 27:0.18159 31:0.0177 34:0.02888 38:1 51:1 63:1 132:1 164:1 236:1
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
        #z = parsed["z"]
        return parsed, y

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TFRecordDataset(filenames).map(_parse_fn, num_parallel_calls=10).prefetch(500000)    # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use
    #dataset = dataset.padded_batch(batch_size, padded_shapes=?)   #不定长补齐 截至TF1.8 Batching of padded sparse tensors is not currently supported

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
    #batch_norm_decay = params["batch_norm_decay"]
    #optimizer = params["optimizer"]
    layers = map(int, params["deep_layers"].split(','))
    dropout = map(float, params["dropout"].split(','))
    attention_layers = map(int, params["attention_layers"].split(','))
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
    a_intids    = features['a_intids']                                          # multi-hot
    #{X multi-hot}
    #x_intids    = features['x_intids']
    #x_intvals   = features['x_intvals']

    #------build f(x)------
    with tf.variable_scope("Embedding-layer"):
        common_embs = tf.nn.embedding_lookup(Feat_Emb, feat_ids)                # None * F' * K
        #uac_emb     = tf.multiply(common_embs, feat_vals)
        a_cat_emb   = tf.nn.embedding_lookup(Feat_Emb, a_catids)                # None * K
        a_shop_emb  = tf.nn.embedding_lookup(Feat_Emb, a_shopids)
        a_brand_emb = tf.nn.embedding_lookup(Feat_Emb, a_brandids)
        a_int_emb   = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=a_intids, sp_weights=None, combiner="sum")

    with tf.variable_scope("Field-wise-Pooling-layer", reuse=tf.AUTO_REUSE):
        if FLAGS.attention_pooling:
            def attention_unit(Feat_Emb, sp_ids, sp_weights, a_xx_emb):
                dense_ids = tf.sparse_tensor_to_dense(sp_ids)
                dense_wgt = tf.expand_dims(tf.sparse_tensor_to_dense(sp_weights), axis=-1)                          # None * P * 1
                dense_emb = tf.nn.embedding_lookup(Feat_Emb, dense_ids)                                             # None * P * K
                dense_emb = tf.multiply(dense_emb, dense_wgt)
                dense_mask = tf.expand_dims(tf.cast(dense_ids > 0, tf.float32), axis=-1)                            # None * P * 1     0=padding >0取非padding
                #dense_mask = tf.sequence_mask(dense_ids, ?)                                                        # None * P
                padded_dim = tf.shape(dense_ids)[1]
                ub_ebm    = tf.reshape(dense_emb, shape=[-1, embedding_size])
                ax_emb    = tf.reshape(tf.tile(a_xx_emb,[1, padded_dim]), shape=[-1, embedding_size])               # None * K --> (None * P) * K     注意跟dense_emb reshape顺序保持一致
                x_inputs  = tf.concat([ub_ebm, ub_ebm - ax_emb, ax_emb], axis=1) 		                            # (None * P) * 3K
                for i in range(len(attention_layers)):
                    x_inputs = tf.contrib.layers.fully_connected(inputs=x_inputs, num_outputs=layers[i], scope='att_fc%d' % i)
                    if FLAGS.batch_norm:
                        x_inputs = batch_norm_layer(x_inputs, train_phase=train_phase, scope_bn='att_bn_%d' %i)
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        x_inputs = tf.nn.dropout(x_inputs, keep_prob=dropout[i])
                att_wgt = tf.contrib.layers.fully_connected(inputs=x_inputs, num_outputs=1, activation_fn=tf.sigmoid, scope='att_out')    #(None * P) * 1
                att_wgt = tf.reshape(att_wgt, shape=[-1, padded_dim, 1])                                            # None * P * 1
                wgt_emb = tf.multiply(dense_emb, att_wgt)                                                           # None * P * K
                wgt_emb = tf.reduce_sum(tf.multiply(wgt_emb, dense_mask), 1) 				                        # None * K
                return wgt_emb

            u_cat_emb   = attention_unit(Feat_Emb, u_catids,   u_catvals,   a_cat_emb)
            u_shop_emb  = attention_unit(Feat_Emb, u_shopids,  u_shopvals,  a_shop_emb)
            u_brand_emb = attention_unit(Feat_Emb, u_brandids, u_brandvals, a_brand_emb)
            u_int_emb   = attention_unit(Feat_Emb, u_intids,   u_intvals,   a_int_emb)
        else:
            u_cat_emb   = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_catids,  sp_weights=u_catvals,   combiner="sum")               # None * K
            u_shop_emb  = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_shopids, sp_weights=u_shopvals,  combiner="sum")
            u_brand_emb = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_brandids,sp_weights=u_brandvals, combiner="sum")
            u_int_emb   = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_intids,  sp_weights=u_intvals,   combiner="sum")

    with tf.variable_scope("MLP-layer"):
        if FLAGS.batch_norm:
            #normalizer_fn = tf.contrib.layers.batch_norm
            #normalizer_fn = tf.layers.batch_normalization
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
                #normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': True, 'reuse': None}
            else:
                train_phase = False
                #normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': False, 'reuse': True}
        else:
            normalizer_fn = None
            normalizer_params = None

        x_deep = tf.concat([tf.reshape(common_embs,shape=[-1, common_dims]),u_cat_emb,u_shop_emb,u_brand_emb,u_int_emb,a_cat_emb,a_shop_emb,a_brand_emb,a_int_emb],axis=1) # None * (F*K)
        for i in range(len(layers)):
            #if FLAGS.batch_norm:
            #    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)
                #normalizer_params.update({'scope': 'bn_%d' %i})
            x_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=layers[i], weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)
            if FLAGS.batch_norm:
                x_deep = batch_norm_layer(x_deep, train_phase=train_phase, scope_bn='bn_%d' %i)   #放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
            if mode == tf.estimator.ModeKeys.TRAIN:
                x_deep = tf.nn.dropout(x_deep, keep_prob=dropout[i])                              #Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)

    with tf.variable_scope("DIN-out"):
        y_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=1, activation_fn=tf.identity, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='din_out')
        y = tf.reshape(y_deep,shape=[-1])
        pred = tf.sigmoid(y)

    predictions={"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + l2_reg * tf.nn.l2_loss(Feat_Emb)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
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
    print('attention_pooling', FLAGS.attention_pooling)
    print('attention_layers', FLAGS.attention_layers)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('batch_norm_decay ', FLAGS.batch_norm_decay)
    print('batch_norm ', FLAGS.batch_norm)
    print('l2_reg ', FLAGS.l2_reg)

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
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout,
        "attention_layers": FLAGS.attention_layers
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
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        #feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        feature_spec = {
            'feat_ids': tf.FixedLenFeature(dtype=tf.int64, shape=[None, FLAGS.field_size]),
            'feat_vals': tf.FixedLenFeature(dtype=tf.float32, shape=[None, FLAGS.field_size])
        }
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        #feature_spec = {
        #    'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
        #    'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        #}
        #serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        Estimator.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
