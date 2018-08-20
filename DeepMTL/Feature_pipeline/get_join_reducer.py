#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" make train date set
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from collections import defaultdict

def doParseLog():
    """parse log on hdfs"""
    #recv = 0
    common_logs = defaultdict(lambda: '')
    sample_list = []
    for line in sys.stdin:
        #recv = recv + 1
        try:
            common_feature_index, log_type, fstrs = line.strip().split('\t')
            if log_type == 'sample':
                sample_list.append(line)
            elif log_type == 'common':
                common_logs[common_feature_index] = fstrs
        except:
            continue

    for sample in sample_list:
        try:
            common_feature_index, _, sample_str = sample.strip().split('\t')
            common_str = common_logs.get(common_feature_index)
            if common_str:
                print "{0} {1}".format(sample_str, common_str)
            else:
                print "{0}".format(sample_str)
        except:
            continue
    #print recv

if __name__ == '__main__':
    doParseLog()
