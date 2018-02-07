# TensorFlow Coustom Estimator Implementation of DeepFM/wide_n_deep/NFM/AFM/FNN/PNN

This repository contains the demo code of DeepFM/wide_n_deep/NFM/AFM/FNN/PNN with the fellowing features：
* Input pipline using Dataset high level API, support parallel and prefetch reading
* Train pipline using Coustom Estimator by rewriting model_fn
* Support distincted training by TF_CONFIG
* Support export model for TensorFlow Serving

## How to use

### feature pipline
We used Criteo dataset. This dataset was used for the Display Advertising Challenge (https://www.kaggle.com/c/criteo-display-ad-challenge).
This code referenced from [here](https://github.com/PaddlePaddle/models/blob/develop/deep_fm/preprocess.py)
There are 13 integer features and 26 categorical features
-For numerical features, normalzied to continous values.
-For categorical features, removed long-tailed data appearing less than 200 times.
After one-hot encoding, the feature space is 117581. Nagetive down sampling will be tryed later.
  python get_criteo_feature.py --input_dir=/dockerdata/lambdaji/ml_packages/data/criteo/ --output_dir=/dockerdata/lambdaji/ml_packages/data/criteo/ --cutoff=200

### model pipline
In practice, these categorical features are usually one-hot encoded for training.
However, this representation results in sparsity.
``train``:
python DeepFM.py --task_type=train --learning_rate=0.0005 --optimizer=Adam --num_epochs=1 --batch_size=256 --field_size=39 --feature_size=117581 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 --num_threads=8 --model_dir=/dockerdata/lambdaji/ml_packages/tf_repos/deep_ctr/model_ckpt/criteo/DeepFM/ --data_dir=/dockerdata/lambdaji/ml_packages/data/criteo/
``infer``:
python DeepFM.py --task_type=infer --learning_rate=0.0005 --optimizer=Adam --num_epochs=1 --batch_size=256 --field_size=39 --feature_size=117581 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 --num_threads=8 --model_dir=/dockerdata/lambdaji/ml_packages/tf_repos/deep_ctr/model_ckpt/criteo/DeepFM/ --data_dir=/dockerdata/lambdaji/ml_packages/data/criteo/

### tf serving pipline
``export``:
python DeepFM.py --task_type=export --learning_rate=0.0005 --optimizer=Adam --batch_size=256 --field_size=39 --feature_size=117581 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 --num_threads=8 --model_dir=/dockerdata/lambdaji/ml_packages/tf_repos/deep_ctr/model_ckpt/criteo/DeepFM/ --servable_model_dir=/dockerdata/lambdaji/ml_packages/tf_repos/deep_ctr/servable_model/

However, our recent experiments and analysis shows that,
training categorical data with simple MLP is difficult.
For this reason, we propose product-nets as the author's first attempt of building new DNN architecture on this field.
This paper has been accepted by ICM2016,
and we will release an extended version very soon.

Discussion about features, models, and training are welcomed,
please contact [Yanru Qu](http://apex.sjtu.edu.cn/members/kevinqu@apexlab.org).

## Environments
Tensorflow (version: 1.0.1)
numpy
sklearn


## How to Use

For simplicity, we provide iPinYou dataset at [make-ipinyou-data](https://github.com/Atomu2014/make-ipinyou-data).
Follow the instructions and update the soft link `data`:

```
XXX/product-nets$ ln -sfn XXX/make-ipinyou-data/2997 data
```

run ``main.py``:

    cd python
    python main.py

As for dataset, we build a repository on github serving as a benchmark in our Lab
[APEX-Datasets](https://github.com/Atomu2014/Ads-RecSys-Datasets).
This repository contains detailed data processing, feature engineering,
data storage/buffering/access and other implementations.
For better I/O performance, this benchmark provides hdf5 APIs.
Currently we provide download links of two large scale ad-click datasets (already processed),
iPinYou and Criteo. Movielens, Netflix, and Yahoo Music will be updated later.

This code is originally written in python 2.7, numpy, scipy and tensorflow are required.
In recent update, we make it consistent with python 3.x.
Thus you can use it as a start-up with any python version you like.
LR, FM, FNN, CCPM and PNN are all implemented in `models.py`, based on TensorFlow.
You can train any of the models in `main.py` and configure parameters via a dict.

More models and mxnet implementation will be released in the extended version.

## Practical Issues

In this section we select some discussions from my email to share.

### 1. Sparse Regularization (L2)

L2 is fundamental in controlling over-fitting.
For sparse input, we suggest sparse regularization,
i.e. we only regularize on activated weights/neurons.

Traditional L2 regularization penalizes all parameters ``w1, .., wn`` even though the input ``xi = 0``,
which means every parameter will have non-zero gradients for every training example.
This is neither reasonable nor efficient for sparse input,
because ``wi*xi = 0`` does not contribute to prediction.

Sparse regularization instead penalizes on non-zero terms, ``xw``.

### 2. Initialization

Initializing weights with small random numbers is always promising in Deep Learning.
Usually we use ``uniform`` or ``normal`` distribution around 0.
An empirical choice is to set the distribution variance near ``sqrt(1/n)`` where n is the input dimension.

Another choice is ``xavier``, for uniform distribution,
``xavier`` uses ``sqrt(3/node_in)``, ``sqrt(3/node_out)``,
or ``sqrt(6/(node_in+node_out))`` as the upper/lower bound.
This is to keep unit variance among different layers.

### 3. Learning Rate

For deep neural networks with a lot of parameters,
large learning rate always causes divergence.
Usually sgd with small learning rate has promising performance, however converges slow.

For extremely sparse input, adaptive learning rate converges much faster,
e.g. AdaGrad, Adam, FTRL, etc.
[This blog](http://sebastianruder.com/optimizing-gradient-descent/)
compares most of adaptive algorithms,
and currently ``adam`` is an empirically good choice.

Even though adaptive algorithms speed up and sometimes jump out of local minimum,
there is no guarantee for better performance because they do not follow gradients' directions.
Some experiments declare that ``adam``'s performance is slightly worse than SGD on some problems.
And sometimes batch size also affect convergence.

### 4. Data Processing

Usually you need to build an index-map to convert categorical data into one-hot representation.
These features usually follow a long-tailed distribution,
resulting in extremely large feature space, e.g. IP.

A simple way is to drop those rarely appearing features by a threshold,
which will dramatically reduce the input dimension without much decrease of performance.

For unbalance dataset, a typical positive/negative ratio is 0.1% - 1%,
and Facebook has published a paper discussing this problem.
Keeping pos and neg samples at similar level is a good choice,
this can be achieved by negative down-sampling.
Negative down-sampling can speed up training, as well as reduce dimension.

### 5. Normalization

There are two kinds of normalization, feature level and instance level.
Feature level is within one field,
e.g. set the mean of one field to 0 and the variance to 1.
Instance level is to keep consistent between difference records,
e.g. you have a multi-value field, which has 5-100 values and the length varies.
You can set the magnitude to 1 by shifting and scaling.
Whitening is not always possible, and normalization is just enough.

Besides, ``batch/weight/layer normalization`` are worth to try when going deeper.

### 6. Continuous/Discrete/Multi-value Feature

Most features in User Response Prediction have discrete values (categorical features).
The key difference between continuous and discrete features is,
discrete features only have absolute meanings (exists or not),
while continuous features have both absolute and relative meanings (higher or smaller).

For example, {'male': 0, 'female': 1} and {'male': 1, 'female': 0} are equivalent,
while numbers 1, 2, 3 can not be arbitrarily encoded.

When your data contains both continuous and discrete values, there arises another problem.
Simply put these features together results in your input neurons having different meanings.
One solution is to discretize those continuous values using bucketing.
Taking 'age' as an example, you can set [0, 12] as 'children', [13, 18] as 'teenagers', [19, ~] as 'adults' as so on.

Multi-value features are special cases of discrete features.
e.g. recently reviewed items = ['item2', 'item7', 'item11'], ['item1', 'item4', 'item9', 'item13'].
Suppose one user has reviewed 3 items, and another has reviewed 300 items,
matrix-multiplication operation will sum these items up and result in huge imbalance.
You may turn to data normalization to tackle this problem.
Till now, there is still not a standard representation for multi-value features,
and we are still working on it.

### 7. Embedding Strategy

In auto-encoder, we may use multi-layer non-linearity to produce low-dim representation,
however in NLP, embedding refers to single-layer linear projection.
Here gives some discussion:

Suppose the embedding layer is followed by a fully connected layer.
The embedding can be represented by ``xw1``.
The fc layer can be represented by ``s(xw1) w2``.
If ``s`` is some nonlinear activation, it can be viewed as two fc layers.
If ``s`` is identity function, it is equivalent to ``xw1 w2 = xw3``.
It seems that you are using ``w3`` as the embedding dimension followed by a nonlinear activation.
The only difference is ``w1 w2`` is a low-rank representation of ``w3``,
but ``w3`` does not have this constraint.

### 8. Activation Function

Do not use ``sigmoid`` in hidden layers, use ``tanh`` or ``relu``.
And recently ``selu`` is proposed to maintain fixed point in training.
