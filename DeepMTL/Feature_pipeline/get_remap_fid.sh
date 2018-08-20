#!/bin/bash
#source src/global_conf.sh
HADOOP_ROOT=/user/mtt/search_mining
HADOOP_BIN=/data/home/hadoop/bin/hadoop
HADOOP_STREAMMING=/data/home/hadoop/hadoop-streaming.jar

# tr te
if [ $# -eq 1 ];then
	task_type=$1
else
	task_type=tr
fi

echo "remap_sample on ${HADOOP_ROOT}/lambdaji/aliccp/${task_type}/sample ..."

INPUT_PATH=${HADOOP_ROOT}/lambdaji/aliccp/${task_type}/sample/*
OUTPUT_PATH=${HADOOP_ROOT}/lambdaji/aliccp/${task_type}/remap_sample

${HADOOP_BIN} fs -rm -r ${OUTPUT_PATH}/

${HADOOP_BIN} jar ${HADOOP_STREAMMING} \
-input ${INPUT_PATH} \
-output ${OUTPUT_PATH} \
-mapper "python get_remap_mapper.py Feat_cnts/feat_cnts" \
-reducer "cat" \
-file "get_remap_mapper.py" \
-cacheArchive "${HADOOP_ROOT}/lambdaji/aliccp/feat_cnts.tar.gz#Feat_cnts" \
-jobconf mapreduce.job.priority=HIGH \
-jobconf mapreduce.map.memory.mb=8192 \
-jobconf mapreduce.map.java.opts=-Xmx8000m \
-jobconf mapreduce.reduce.memory.mb=8192 \
-jobconf mapreduce.reduce.java.opts=-Xmx8000m \
-jobconf mapred.map.capacity.per.tasktracker=3 \
-jobconf mapred.reduce.capacity.per.tasktracker=3 \
-jobconf mapred.task.timeout=7200000 \
-jobconf mapreduce.job.maps=500 \
-jobconf mapreduce.job.reduces=100 \
-jobconf mapreduce.job.queuename=root.mtt.default \
-jobconf mapreduce.job.name="t_sd_mtt_aliccp_stat_${task_type}_lambdaji"

if [ ${?} -eq 0 ]
then
    echo "succeed"
else
    echo "failed"
    exit 1
fi

echo "remap_sample on ${HADOOP_ROOT}/lambdaji/aliccp/${task_type}/sample stat:${?}"
