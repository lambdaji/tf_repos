## TensorFlow Estimator of DeepCTR - DeepFM/wide_n_deep/FNN/PNN/NFM/AFM/DCN
深度学习在ctr预估领域的应用越来越多，新的模型结构层出不穷。但是这些模型如何验证是否有效，快速在工业界落地仍然存在一些问题：

* 开源的实现基本都是学术界的人在搞，距离工业应用还有较大的鸿沟
* 模型实现大量调用底层且低版本的API，兼容和性能很成问题
* 单机，放到工业场景下跑不动
* 迁移成本较高，每种算法都要从头开始搭一套

针对存在的问题做了一些探索，摸索出一套基于TensorFlow的统一框架，有以下特性：

* 读数据采用Dataset API，支持 parallel and prefetch读取
* 通过Estimator封装算法f(x)，实验新算法边际成本比较低，只需要改写model_fn f(x)部分
* 支持分布式以及单机多线程训练
* 支持export model，然后用TensorFlow Serving提供线上预测服务

[我的知乎](https://zhuanlan.zhihu.com/p/33699909)

## How to use
pipline: feature → model → serving

### 特征框架 -- logs in，samples out
实验数据集用criteo，特征工程参考[here](https://github.com/PaddlePaddle/models/blob/develop/deep_fm/preprocess.py)

DNN做ctr预估的优势在于对大规模离散特征建模，paper关注点大都放在ID类特征如何做embedding上，至于连续特征如何处理很少讨论，大概有以下3种方式：

    --不做embedding
      |1--concat[continuous, emb_vec]做fc
    --做embedding
      |2--离散化之后embedding
      |3--类似FM二阶部分, 统一做embedding, <id, val> 离散特征val=1.0
为了模型设计上的简单统一，采用第3种方式，感兴趣的同学可以试试前两种的效果。

    python get_criteo_feature.py --input_dir=../../data/criteo/ --output_dir=../../data/criteo/ --cutoff=200

### 训练框架 -- samples in，model out
用Tensorflow (version: 1.4)作为训练框架，目前实现了DeepFM/wide_n_deep/FNN/PNN/NFM/AFM/Deep & Cross Network等算法，除了wide_n_deep，其他算法默认参数.

![tensorboard_auc.png](https://github.com/lambdaji/tf_repos/raw/master/deep_ctr/uploads/tensorboard_auc.png)

其实很多模型结构的表达能力都是同等的，性能上的差异主要是由于某些结构比其他结构更容易优化导致的。下图是固定同一组超参比较不同网络结构的特性，可以发现：
- FNN/Inner-PNN/DeepFM/DCN几个算法AUC=0.8±0.003
- 结构相对简单的FNN收敛较慢，但是AUC能够达到0.8!
- DeepFM&DCN通过shallow part把反馈信号传回embedding层，收敛得更快。

![same_hyper.png](https://github.com/lambdaji/tf_repos/raw/master/deep_ctr/uploads/same_hyper.png)

以DeepFM为例来看看如何使用：

``train``:

    python DeepFM.py --task_type=train --learning_rate=0.0005 --optimizer=Adam --num_epochs=1 --batch_size=256 --field_size=39 --feature_size=117581 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 --num_threads=8 --model_dir=./model_ckpt/criteo/DeepFM/ --data_dir=../../data/criteo/

``infer``:

    python DeepFM.py --task_type=infer --learning_rate=0.0005 --optimizer=Adam --num_epochs=1 --batch_size=256 --field_size=39 --feature_size=117581 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 --num_threads=8 --model_dir=./model_ckpt/criteo/DeepFM/ --data_dir=../../data/criteo/

### 服务框架 -- request in，pctr out
线上预测服务使用TensorFlow Serving+TAF搭建。TensorFlow Serving是一个用于机器学习模型 serving 的高性能开源库，使用 gRPC 作为接口接受外部调用，它支持模型热更新与自动模型版本管理。
首先要导出TF-Serving能识别的模型文件：

    python DeepFM.py --task_type=export --learning_rate=0.0005 --optimizer=Adam --batch_size=256 --field_size=39 --feature_size=117581 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 --num_threads=8 --model_dir=./model_ckpt/criteo/DeepFM/ --servable_model_dir=./servable_model/


默认以时间戳来管理版本，生成文件如下：

      $ ls -lh servable_model/1517971230
      |--saved_model.pb
      |--variables
        |--variables.data-00000-of-00001
        |--variables.index

然后写client发送请求，参考Serving_pipeline。

wide_n_deep model线上预测性能如下：

![tf_serving_wdl.png](https://github.com/lambdaji/tf_repos/raw/master/deep_ctr/uploads/tf_serving_wdl.png)

可以看到：

    截距部分15ms：对应解析请求包，查询DCache，转换特征格式以及打log等
    斜率部分0.5ms：大约一条样本forward一次需要的时间

### 参考资料
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

[2] Wide & Deep Learning for Recommender Systems

[3] Deep Learning over Multi-Field Categorical Data: A Case Study on User Response Prediction

[4] Product-based Neural Networks for User Response Prediction

[5] Deep & Cross Network for Ad Click Predictions

[6] Deep Interest Network for Click-Through Rate Prediction

[7] Neural Factorization Machines for Sparse Predictive Analytics

[8] Attentional Factorization Machines:Learning theWeight of Feature Interactions via Attention Networks

[9] https://github.com/Atomu2014/product-nets

[10] https://github.com/hexiangnan/attentional_factorization_machine

[11] https://github.com/hexiangnan/neural_factorization_machine
