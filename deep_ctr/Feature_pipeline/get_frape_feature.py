#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
label map to {0, 1}
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from multiprocessing import Pool as ThreadPool
import argparse
import glob
from collections import defaultdict

#-1 451:1 4149:1 5041:1 5046:1 5053:1 5055:1 5058:1 5060:1 5069:1 5149:1

def get_frape_feature(if_str):
    output = open(if_str.split('.')[0] +'_.libsvm','w')
    num = 0
    with open(if_str,"rt") as f:
        for line in f:
            try:
                label, feats = line.strip().split(' ', 1)
                if label == '-1':
                    label = '0'

                output.write("{0} {1}\n".format(label, feats))
            except:
                #output.write(line)
                continue

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
        default="",
        help="input data dir"
        )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="feature map output dir"
        )


    FLAGS, unparsed = parser.parse_known_args()
    print('threads ', FLAGS.threads)
    print('input_dir ', FLAGS.input_dir)
    print('output_dir ', FLAGS.output_dir)

    file_list = glob.glob(FLAGS.input_dir+'/*libsvm')
    print('file_list size ', len(file_list))
    pool = ThreadPool(FLAGS.threads) # Sets the pool size
    pool.map(get_frape_feature, file_list)
    pool.close()
    pool.join()
