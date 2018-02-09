#coding=utf8
"""
This code referenced from [here](https://github.com/PaddlePaddle/models/blob/develop/deep_fm/preprocess.py)
-For numerical features,normalzied to continous values.
-For categorical features, removed long-tailed data appearing less than 200 times.

TODO：
#1 连续特征 离散化
#2 Nagetive down sampling
"""
import os
import sys
#import click
import random
import collections
import argparse
from multiprocessing import Pool as ThreadPool

# There are 13 integer features and 26 categorical features
continous_features = range(1, 14)
categorial_features = range(14, 40)

# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return map(len, self.dicts)


class ContinuousFeatureGenerator:
    """
    Normalize the integer features to [0, 1] by min-max normalization
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxint] * num_feature
        self.max = [-sys.maxint] * num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > continous_clip[i]:
                            val = continous_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])


#@click.command("preprocess")
#@click.option("--datadir", type=str, help="Path to raw criteo dataset")
#@click.option("--outdir", type=str, help="Path to save the processed data")
def preprocess(datadir, outdir):
    """
    All the 13 integer features are normalzied to continous values and these
    continous features are combined into one vecotr with dimension 13.
    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
    """
    #pool = ThreadPool(FLAGS.threads) # Sets the pool size
    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(FLAGS.input_dir + 'train.txt', continous_features)
    #pool.apply(dists.build, args=(FLAGS.input_dir + 'train.txt', continous_features,))

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(FLAGS.input_dir + 'train.txt', categorial_features, cutoff=FLAGS.cutoff)
    #pool.apply(dicts.build, args=(FLAGS.input_dir + 'train.txt', categorial_features,))

    #pool.close()
    #pool.join()

    output = open(FLAGS.output_dir + 'feature_map','w')
    for i in continous_features:
        output.write("{0} {1}\n".format('I'+str(i), i))
    dict_sizes = dicts.dicts_sizes()
    categorial_feature_offset = [dists.num_feature]
    for i in range(1, len(categorial_features)+1):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)
        for key, val in dicts.dicts[i-1].iteritems():
            output.write("{0} {1}\n".format('C'+str(i)+'|'+key, categorial_feature_offset[i - 1]+val+1))

    random.seed(0)

    # 90% of the data are used for training, and 10% of the data are used
    # for validation.
    with open(FLAGS.output_dir + 'tr.libsvm', 'w') as out_train:
        with open(FLAGS.output_dir + 'va.libsvm', 'w') as out_valid:
            with open(FLAGS.input_dir + 'train.txt', 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')

                    feat_vals = []
                    for i in range(0, len(continous_features)):
                        val = dists.gen(i, features[continous_features[i]])
                        feat_vals.append(str(continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                    for i in range(0, len(categorial_features)):
                        val = dicts.gen(i, features[categorial_features[i]]) + categorial_feature_offset[i]
                        feat_vals.append(str(val) + ':1')

                    label = features[0]
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write("{0} {1}\n".format(label, ' '.join(feat_vals)))
                    else:
                        out_valid.write("{0} {1}\n".format(label, ' '.join(feat_vals)))

    with open(FLAGS.output_dir + 'te.libsvm', 'w') as out:
        with open(FLAGS.input_dir + 'test.txt', 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                feat_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    feat_vals.append(str(continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i] - 1]) + categorial_feature_offset[i]
                    feat_vals.append(str(val) + ':1')

                out.write("{0} {1}\n".format(label, ' '.join(feat_vals)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="threads num"
        )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="input data dir"
        )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="feature map output dir"
        )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=200,
        help="cutoff long-tailed categorical values"
        )

    FLAGS, unparsed = parser.parse_known_args()
    print('threads ', FLAGS.threads)
    print('input_dir ', FLAGS.input_dir)
    print('output_dir ', FLAGS.output_dir)
    print('cutoff ', FLAGS.cutoff)

    preprocess(FLAGS.input_dir, FLAGS.output_dir)
