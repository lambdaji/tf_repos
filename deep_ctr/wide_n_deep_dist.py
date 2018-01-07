#!/usr/bin/env python
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import os
import glob
import json
import threading
import random
from datetime import date, timedelta

import numpy as np
import tensorflow as tf

# distributed flags
FLAGS = tf.app.flags.FLAGS
LOG = tf.logging
tf.app.flags.DEFINE_boolean("dist_mode", False, "run use distribuion mode or not")
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("task_type","train_eval","{'train_eval', 'predict', 'export_model'}")
tf.app.flags.DEFINE_string("model_type","deep","{'wide', 'deep', 'wide_n_deep'}")
tf.app.flags.DEFINE_string("model_dir","/data/ceph/smartbox/model_ckpt/","base dir for output model ckpt")
tf.app.flags.DEFINE_string("servable_model_dir","/data/ceph/smartbox/servable_model/","base dir for output model ckpt")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")
tf.app.flags.DEFINE_string("log_dir","/data/ceph/smartbox/log/","directoriy for saving logs for Tensorboard")
tf.app.flags.DEFINE_integer("num_threads", 10, "number of training epochs")
tf.app.flags.DEFINE_integer("train_epochs", 100, "number of training epochs")
tf.app.flags.DEFINE_integer("eval_steps", 100000, "number of training epochs")
tf.app.flags.DEFINE_string("data_dir","/data/ceph/smartbox/data/","directoriy for training data")
tf.app.flags.DEFINE_string("dt_dir","","training data date partition")
tf.app.flags.DEFINE_string("pred_dir","/data/ceph/smartbox/data/","path of data to predict")
tf.app.flags.DEFINE_integer("embedding_size", 32, "embedding size")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size for training")
tf.app.flags.DEFINE_string("hidden_layer", "256,128,64", "hidden unit size")
tf.app.flags.DEFINE_string("test_file", "./wdl_te.csv", "test file")
tf.app.flags.DEFINE_string("pred_file", "./predict.csv", "predict result file")

###############################################################################
#
#       { < u, a, c, xgb >, y }
#
################################################################################
#+------------------------------+
#+     Initialization           +
#+------------------------------+
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
print(CSV_COLUMN_DEFAULTS)

#DISCRETE_COLUMNS
#"u_gender", "u_age", "u_city", "u_constell", "u_education", "u_mobile", "a_adtype", "a_adpos", "c_q_from", "c_q_len", "c_q_ch_en"
D_COLUMNS = [ "u_pl","u_ppvn","u_de","u_os","u_t","a_m_w","a_b_w","c_h","c_w","c_al" ]

#CONTINUOUS_COLUMNS
#c_q_t_lda c_q_t_w2v c_q_t_cla
C_COLUMNS = [ "u_ctr","a_a_ctr","a_t_ctr","c_q_ctr","c_al_ctr","c_n_ctr","c_t_ctr","c_t_n_ctr",
              "u_a_city_ctr","u_a_age_ctr","u_a_x_ctr","u_a_g_ctr","u_a_c_ctr","c_q_a_ctr","c_q_t_sim","c_q_adtype_ctr","c_mw_a_ctr" ]

def build_feature():
    #+------------------------------+
    #+    Discrete base columns     +
    #+------------------------------+
    #1 { user feture }
    #gender = tf.feature_column.categorical_column_with_identity(key='gender', num_buckets=3, default_value=0)
    #age = tf.feature_column.categorical_column_with_identity(key='age', num_buckets=7, default_value=0)
    #city = tf.feature_column.categorical_column_with_identity(key='city', num_buckets=7, default_value=0)
    #education = tf.feature_column.categorical_column_with_identity(key='education', num_buckets=9, default_value=0)
    #constell = tf.feature_column.categorical_column_with_identity(key='constell', num_buckets=13, default_value=0)
    #news_topics = tf.feature_column.categorical_column_with_hash_bucket(key="news_topics", hash_bucket_size=500)
    platform = tf.feature_column.categorical_column_with_vocabulary_list("u_pl", ["ADR", "IOS"])
    ppvn = tf.feature_column.categorical_column_with_hash_bucket(key="u_ppvn", hash_bucket_size=20)
    device = tf.feature_column.categorical_column_with_vocabulary_list("u_de", ["PAD", "PHONE"])
    #mobile = tf.feature_column.categorical_column_with_hash_bucket(key="mobile", hash_bucket_size=8000)
    #rl = tf.feature_column.categorical_column_with_hash_bucket(key="u_rl", hash_bucket_size=500)
    os = tf.feature_column.categorical_column_with_hash_bucket(key="u_os", hash_bucket_size=60)
    tele_service = tf.feature_column.categorical_column_with_vocabulary_list("u_t", ['0', '1', '3', '20', '21'])

    #2 { item feture }
    #ad_type = tf.feature_column.categorical_column_with_identity(key="ad_type", num_buckets=100, default_value=0)
    #ad_pos = tf.feature_column.categorical_column_with_vocabulary_list("ad_pos", [0, 1, 2])
    mark_word = tf.feature_column.categorical_column_with_hash_bucket(key="a_m_w", hash_bucket_size=60)
    button_word = tf.feature_column.categorical_column_with_hash_bucket(key="a_b_w", hash_bucket_size=20)

    #3 { context feture }
    hour = tf.feature_column.categorical_column_with_hash_bucket(key="c_h", hash_bucket_size=24)
    week = tf.feature_column.categorical_column_with_hash_bucket(key="c_w", hash_bucket_size=7)
    #query_from = tf.feature_column.categorical_column_with_vocabulary_list("query_from", [0, 1, 2])
    #query_len = tf.feature_column.categorical_column_with_identity(key="query_len", num_buckets=25, default_value=25)
    #query_type = tf.feature_column.categorical_column_with_vocabulary_list("query_from", ["ch", "en"])
    #mw_pos = tf.feature_column.categorical_column_with_identity(key="c_al", num_buckets=8, default_value=7)
    mw_pos = tf.feature_column.categorical_column_with_hash_bucket(key="c_al", hash_bucket_size=10)

    #wide_dbc = [platform, device, ppvn, os, tele_service, mark_word, button_word, hour, week, mw_pos]
    wide_dbc = []

    #4 { XGB feature }
    #wide_xgb = [tf.feature_column.categorical_column_with_identity(key=col, num_buckets=1024) for col in XGB_COLUMNS]
    wide_xgb = [tf.feature_column.categorical_column_with_hash_bucket(key=col, hash_bucket_size=1024, dtype=tf.int64) for col in XGB_COLUMNS]

    #5 { Crossed feature }  (todo)
    wide_cross = []
    #cross between user and item
    #wide_cross = [
    #    tf.feature_column.crossed_column(
    #        ["city", "ad_type"], hash_bucket_size=1000),
    #    tf.feature_column.crossed_column(
    #        ["news_topics", "ad_type"], hash_bucket_size=1000)
    #]

    #cross between context and item


    #6 { Embedding dbc }
    #ebc = [age, city, constell, education, news_topics, ppvn, os, ad_type, mark_word, button_word]
    ebc = [ppvn, os, mark_word, button_word, platform, device, tele_service, hour, week, mw_pos]
    #ebc = [mark_word, button_word, mw_pos]
    deep_emb = [tf.feature_column.embedding_column(c, dimension=FLAGS.embedding_size) for c in ebc]

    #+------------------------------+
    #+    Continuous base columns   +
    #+------------------------------+
    deep_cbc = [tf.feature_column.numeric_column(colname) for colname in C_COLUMNS]

    #normalization (todo)

    #+-------------------------------+
    #+ Wide columns and deep columns +
    #+-------------------------------+
    wide_columns = wide_dbc + wide_xgb + wide_cross
    deep_columns = deep_cbc + deep_emb

    return wide_columns, deep_columns

def input_fn(filenames, num_epochs, batch_size=1):
    def parse_csv(value):
        print('Parsing', filenames)
        columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        labels = features.pop(LABEL_COLUMN)
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(filenames) # can pass one filename or filename list

    # multi-thread pre-process then prefetch
    dataset = dataset.map(parse_csv, num_parallel_calls=10).prefetch(500000)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels

################################################################################
#
#       f(x) / loss / Optimizer
#
################################################################################
def build_estimator(model_dir, model_type):
    """Build an estimator."""

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % model_dir)

    #build wide_columns, deep_columns
    wide_columns, deep_columns = build_feature()

    hidden_units = map(int, FLAGS.hidden_layer.split(","))
    config = tf.estimator.RunConfig().replace(
            session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
            save_checkpoints_steps=FLAGS.eval_steps,
            log_step_count_steps=1000,
            save_summary_steps=1000)
    #bulid model
    if model_type == "wide":
        estimator = tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=config)
    elif model_type == "deep":
        estimator = tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=config)
    else:
        estimator = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=config)

    return estimator

def train_and_eval(model_dir, model_type, train_data):
    """Train and evaluate the model."""
    # build model
    estimator = build_estimator(model_dir, model_type)

    # shuffle traning files order for preventing overfittng
    file_list = glob.glob("%s/wdl_train_part*" % train_data)
    random.shuffle(file_list)
    LOG.info(file_list)
    LOG.info("file_list size =  %d\n" % len(file_list))
    eval_fin = train_data + "/" + "wdl_merge.test"

    wide_columns, deep_columns = build_feature()
    if model_type == "deep":
        feature_columns = deep_columns
    elif model_type == "wide":
        feature_columns = wide_columns
    elif model_type == "wide_n_deep":
        feature_columns = wide_columns + deep_columns

    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    lastest_exporter = tf.estimator.LatestExporter(
            model_type,
            serving_input_receiver_fn,
            exports_to_keep=None)
    if FLAGS.dist_mode:
        worker_id = int(FLAGS.task_index)
        # 减2表示去掉chief结点和evaluator结点
        num_workers = len(FLAGS.worker_hosts.split(',')) - 2
        LOG.info("worker_id = %d/%d" % (worker_id, num_workers))

    train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(file_list, num_epochs=FLAGS.train_epochs, batch_size=FLAGS.batch_size))
    eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(eval_fin, num_epochs=1, batch_size=FLAGS.batch_size),
            exporters = lastest_exporter,
            steps = None,  # evaluate the whole eval file
            start_delay_secs=3000,
            throttle_secs=3600) # evaluate every one hour
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def predict(model_dir, model_type, test_file, pred_file):
    """predict the results for an input dataset."""
    #+------------------------------+
    #+ restore model from model_dir +
    #+------------------------------+
    # m = build_estimator(model_dir, model_type)
    #build wide_columns, deep_columns
    wide_columns, deep_columns = build_feature()

    # build model
    estimator = build_estimator(model_dir, model_type)

    pred = estimator.predict(input_fn=lambda: input_fn(test_file, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="probabilities")

    prob_result = []
    pred_cnt = 0
    with open(pred_file, "w") as fo:
        for prob in pred:
            pred_cnt += 1
            if pred_cnt % 10000:
                print("predict %d result" % (pred_cnt))
            fo.write("%f\n" % (prob['probabilities'][1]))

def export_serving_model(model_dir, model_type, servable_model_dir):
    """Export serving model."""
    # rebuild model
    wide_columns, deep_columns = build_feature()
    estimator = build_estimator(model_dir, model_type)
    feature_columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    #serving_input_receiver_fn = tf.contrib.learn.build_parsing_serving_input_fn(feature_spec)
    estimator.export_savedmodel(servable_model_dir, serving_input_receiver_fn)


def main(_):
    dt_dir = FLAGS.dt_dir
    FLAGS.model_dir = FLAGS.model_dir
    FLAGS.data_dir  = FLAGS.data_dir
    FLAGS.pred_dir  = FLAGS.pred_dir
    FLAGS.log_dir   = FLAGS.log_dir  + dt_dir
    FLAGS.servable_model_dir = FLAGS.servable_model_dir + dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_type ', FLAGS.model_type)
    print('model_dir ', FLAGS.model_dir)
    print('servable_model_dir ', FLAGS.servable_model_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('train_data ', FLAGS.data_dir)
    print('log_dir ', FLAGS.log_dir)
    print('train_epochs ', FLAGS.train_epochs)
    print('embedding_size ', FLAGS.embedding_size)
    print('hidden_layer ', FLAGS.hidden_layer)
    print('batch_size ', FLAGS.batch_size)
    if FLAGS.task_type == "train_eval":
        if FLAGS.dist_mode:
            ps_hosts = FLAGS.ps_hosts.split(',')
            worker_hosts = FLAGS.worker_hosts.split(',')
            chief_hosts = worker_hosts[0:1] # get first worker as chief
            worker_hosts = worker_hosts[2:] # the rest as worker
            print('ps_host', ps_hosts)
            print('worker_host', worker_hosts)
            print('chief_hosts', chief_hosts)
            task_index = FLAGS.task_index
            job_name = FLAGS.job_name
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

            tf_config = {'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts}, 'task': {'type': job_name, 'index': task_index }}
            LOG.info(json.dumps(tf_config))
            os.environ['TF_CONFIG'] = json.dumps(tf_config)

        train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.data_dir)
    elif FLAGS.task_type == "predict":
        predict(FLAGS.model_dir, FLAGS.model_type, FLAGS.test_file, FLAGS.pred_file)
    elif FLAGS.task_type == "export_model":
        export_serving_model(FLAGS.model_dir, FLAGS.model_type, FLAGS.servable_model_dir)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
