#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
bml to csv
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from multiprocessing import Pool as ThreadPool
import argparse
import glob
from collections import defaultdict

#0,ADR,7.0,PHONE,OOV,0,经验,查看,22,Mon,0,0.06808,0.01559,0.01714,0.00524,0.01983,0.01053,0.01708,0.01273,0.01161,0.00794,0.00938,0.00867,0.0086,0.0059,0.07692,0.00577,0.00562,
#1207,1235,1354,1379,1363,1382,1304,1313,1202,1251,1101,1136,970,693,1184,843,980,710,1173,700,766,570,430,732,340,475,684,566,632,460,480,453,390,440,582,334,533,521,346,421,372,770,707,575,470,674,394,530,332,225,583,532,270,510,513,387,701,264,251,496,230,689,678,515,299,569,786,671,321,599,498,324,532,404,259,582,342,232,700,688,468,623,722,627,280,546,327,522,356,188,297,523,378,398,125,202,330,737,603,376

#保证顺序以及数量
#Constants
xgb_trees = 100
# Column Title
CSV_COLUMNS = [ "is_click","u_pl","u_ppvn","u_de","u_os","u_t","a_m_w","a_b_w","c_h","c_w","c_al",
                "u_ctr","a_a_ctr","a_t_ctr","c_q_ctr","c_al_ctr","c_n_ctr","c_t_ctr","c_t_n_ctr",
                "u_a_city_ctr","u_a_age_ctr","u_a_x_ctr","u_a_g_ctr","u_a_c_ctr","c_q_a_ctr","c_q_t_sim","c_q_adtype_ctr","c_mw_a_ctr" ]
XGB_COLUMNS = [ 'xgbf_%d' % i for i in range(xgb_trees) ]
CSV_COLUMNS = CSV_COLUMNS + XGB_COLUMNS

def get_feature_map(file_list):
    output = open(FLAGS.output_dir + 'feature_map','w')
    feature_map = defaultdict(lambda: 0)
    fid = 1
    for fname in CSV_COLUMNS:
        feature_map[CSV_COLUMNS[i] + '|UNK'] = fid
        fid += 1

    for if_str in file_list:
        with open(if_str,"rt") as f:
            for line in f:
                try:
                    splits = line.strip().split(",")
                    for i in range(1,len(splits[1:])):
                        if(i >= 11 and i <= 27):
                            key = CSV_COLUMNS[i]
                        else:
                            key = CSV_COLUMNS[i] + '|' + splits[i]
                        if feature_map.get(key) == None:
                            feature_map[key] = fid
                            fid += 1
                except:
                    #output.write(line)
                    continue

    for key,val in feature_map.iteritems():
        output.write("{0} {1}\n".format(key,val))

def get_smart_feature(if_str):
    feature_map = defaultdict(lambda: 0)
    with open(FLAGS.output_dir + 'feature_map','rt') as f:
        for line in f:
            try:
                splits = line.strip().split(" ")
                feature_map[splits[0]] = splits[1]
            except:
                continue
    prefix = ''
    if FLAGS.task_type == 'tr':
        prefix = prefix + '_' + if_str.rsplit('_')[3]
    output = open(FLAGS.output_dir + FLAGS.task_type + prefix +'.libsvm','w')
    with open(if_str,"rt") as f:
        for line in f:
            try:
                splits = line.strip().split(",")
                is_click = splits[0]
                feat = []
                for i in range(1,len(splits[1:])):
                    if(i >= 11 and i <= 27):
                        key = CSV_COLUMNS[i]
                        fid = feature_map.get(key)
                        feat.append(str(fid) + ':' + splits[i])
                    else:
                        key = CSV_COLUMNS[i] + '|' + splits[i]
                        fid = feature_map.get(key)
                        if fid == None:
                            fid = feature_map.get(CSV_COLUMNS[i] + '|UNK')
                        feat.append(str(fid) + ':1')

                output.write("{0} {1}\n".format(is_click,' '.join(feat)))
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
    parser.add_argument(
        "--task_type",
        type=str,
        default="tr",
        help="{tr,va,te}"
        )

    FLAGS, unparsed = parser.parse_known_args()
    print('threads ', FLAGS.threads)
    print('input_dir ', FLAGS.input_dir)
    print('output_dir ', FLAGS.output_dir)
    print('task_type ', FLAGS.task_type)

    if FLAGS.task_type == 'tr':
        file_list = glob.glob(FLAGS.input_dir+'/*part*')
        #FeatureMapGenerator
        #get_feature_map(file_list)
    elif FLAGS.task_type == 'va':
        file_list = glob.glob(FLAGS.input_dir+'/*verify')
    elif FLAGS.task_type == 'te':
        file_list = glob.glob(FLAGS.input_dir+'/*test')

    print('file_list size ', len(file_list))
    pool = ThreadPool(FLAGS.threads) # Sets the pool size
    pool.map(get_smart_feature, file_list)
    pool.close()
    pool.join()
