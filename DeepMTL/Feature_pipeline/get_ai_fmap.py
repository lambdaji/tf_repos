#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
get feature map
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from multiprocessing import Pool as ThreadPool
import argparse
import glob
from collections import defaultdict

def get_feat_stat(if_str):
    output = open(if_str +'.stat','w')
    feat_dict = defaultdict(lambda: 0)
    with open(if_str,"rt") as f:
        for line in f:
            try:
                splits = line.strip().split('\t')
                filed, feat = ff.split(':')
                field_dict[filed] += int(cnts)
                feat_dict[feat] += int(cnts)
                if int(feat) > max_feat:
                    max_feat = int(feat)
            except:
                #output.write(line)
                continue

    output.write("lines\t{0}\n".format(num))

    output.write("--------------\n")
    output.write("max_feat\t{0}\t{1}\n".format(max_feat,len(feat_dict)))

    output.write("--------------\n")
    for key,val in field_dict.iteritems():
        output.write("{0}\t{1}\n".format(key,val))

    output.write("--------------\n")
    feat_cnts = defaultdict(lambda: 0)
    for key,val in feat_dict.iteritems():
        feat_cnts[val] += 1
        output.write("{0}\t{1}\n".format(key,val))

    out = open(if_str +'.cnts','w')
    for key,val in feat_cnts.iteritems():
        out.write("{0}\t{1}\n".format(key,val))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="threads num"
        )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./",
        help="input data dir"
        )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="feature map output dir"
        )


    FLAGS, unparsed = parser.parse_known_args()
    print('threads ', FLAGS.threads)
    print('input_dir ', FLAGS.input_dir)
    print('output_dir ', FLAGS.output_dir)

    file_list = glob.glob(FLAGS.input_dir+'/feat_map*')
    print('file_list size ', len(file_list))
    pool = ThreadPool(FLAGS.threads) # Sets the pool size
    pool.map(get_feat_stat, file_list)
    pool.close()
    pool.join()
