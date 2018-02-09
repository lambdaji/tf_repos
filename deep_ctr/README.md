# TensorFlow Estimator Implementation of DeepFM/wide_n_deep/NFM/AFM/FNN/PNN

This repository contains the demo code of DeepFM/wide_n_deep/NFM/AFM/FNN/PNN with the fellowing features：
* Input pipline using Dataset high level API, support parallel and prefetch reading
* Train pipline using Coustom Estimator by rewriting model_fn
* Support distincted training by TF_CONFIG and multi-threads
* Support export model for TensorFlow Serving

# Environments
* Tensorflow (version: 1.4)

## How to use

### feature pipline
This dataset was used for the Display Advertising Challenge (https://www.kaggle.com/c/criteo-display-ad-challenge).
There are 13 integer features and 26 categorical features:
-For numerical features, normalzied to continous values.
-For categorical features, removed long-tailed data appearing less than 200 times.
After one-hot encoding, the feature space is 117581. Nagetive down sampling will be tryed later.

This code referenced from [here](https://github.com/PaddlePaddle/models/blob/develop/deep_fm/preprocess.py)

    python get_criteo_feature.py --input_dir=../../data/criteo/ --output_dir=../../data/criteo/ --cutoff=200

### model pipline
``train``:

    python DeepFM.py --task_type=train --learning_rate=0.0005 --optimizer=Adam --num_epochs=1 --batch_size=256 --field_size=39 --feature_size=117581 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 --num_threads=8 --model_dir=./model_ckpt/criteo/DeepFM/ --data_dir=../../data/criteo/

``infer``:

    python DeepFM.py --task_type=infer --learning_rate=0.0005 --optimizer=Adam --num_epochs=1 --batch_size=256 --field_size=39 --feature_size=117581 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 --num_threads=8 --model_dir=./model_ckpt/criteo/DeepFM/ --data_dir=../../data/criteo/

### tf serving pipline
Serving a TensorFlow Estimator model in C++ by TF-Serving. This tutorial consists of two parts:
* Exporting model using .export_savedmodel().
* Creating a client for the model(Input and Output must be matched with ``serving_input_receiver_fn``) and serving it.

``export``:

    python DeepFM.py --task_type=export --learning_rate=0.0005 --optimizer=Adam --batch_size=256 --field_size=39 --feature_size=117581 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 --num_threads=8 --model_dir=./model_ckpt/criteo/DeepFM/ --servable_model_dir=./servable_model/

``servable_model_dir`` will contain the following files:
    $ ls -lh servable_model/1517971230
    |--saved_model.pb
    |--variables
      |--variables.data-00000-of-00001
      |--variables.index

``TF-Serving``:
  Please refer the ``Serving_pipline`` folder.

You'd better take a look at the following proto file first in:
* tensorflow/core/example/example.proto
* tensorflow/core/example/feature.proto
* tensorflow/core/framework/tensor.proto
* tensorflow_serving/apis/predict.proto
* tensorflow_serving/apis/model.proto
