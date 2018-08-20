#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" re map feat_id """
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from collections import defaultdict
import random

def load_fcnts(if_str):
    feat_cnts_dict = defaultdict(lambda: 0)
    new_id = 20
    with open(if_str) as f:
        for line in f:
            fid, cnts = line.strip().split('\t')
            if feat_cnts_dict.get(fid):
                continue
            if int(cnts) >= 20:             #cutoff=20
                feat_cnts_dict[fid] = new_id
                new_id = new_id + 1
    return feat_cnts_dict

def doParseLog(feat_cnts_dict):
    """parse log on hdfs"""
    for line in sys.stdin:
        #recv = recv + 1
        try:
            splits = line.strip().split(',')
            # y=0 & z=1过滤
            if(splits[1] == '0' and splits[2] == '1'):
                continue
            # remap feat_id
            feat_lists = []
            for fstr in splits[3].split(' '):
                f,fid,val = fstr.split(':')
                new_id = feat_cnts_dict.get(fid)
                if new_id:
                    feat_lists.append('%s:%d:%s' % (f,new_id,val))
            ri = random.randint(0, 2147483647)      #shuffle
            print "{0}\t{1},{2},{3},{4}".format(ri, splits[0], splits[1], splits[2], ' '.join(feat_lists))
        except:
            continue

if __name__ == '__main__':
    feat_cnts_file, = sys.argv[1:2]
    feat_cnts_dict = load_fcnts(feat_cnts_file)
    doParseLog(feat_cnts_dict)
