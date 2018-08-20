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
    cnts_dict = defaultdict(lambda: 0)
    for line in sys.stdin:
        #recv = recv + 1
        try:
            key, val = line.strip().split('\t')
            cnts_dict[key] += int(val)
        except:
            continue
    for key,val in cnts_dict.iteritems():
        print "{0}\t{1}".format(key,val)

if __name__ == '__main__':
    doParseLog()
