#!/bin/bash

model_dir=/dockerdata/lambdaji/ml_packages/tf_repos/deep_ctr/model_ckpt/criteo/
data_dir=/dockerdata/lambdaji/ml_packages/data/criteo/

python DeepFM.py --learning_rate=0.0005 --optimizer=Adam --num_epochs=1 \
                 --batch_size=256 --field_size=39 --feature_size=117581 \
                 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 \
                 --num_threads=8 --model_dir=${model_dir}/DeepFM/ --data_dir=${data_dir} \
                 --ps_hosts=localhost:2222 \
                 --worker_hosts=localhost:2223,localhost:2224,localhost:2225 \
                 --job_name=ps --task_index=0 &

sleep 1
python DeepFM.py --learning_rate=0.0005 --optimizer=Adam --num_epochs=1 \
                 --batch_size=256 --field_size=39 --feature_size=117581 \
                 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 \
                 --num_threads=8 --model_dir=${model_dir}/DeepFM/ --data_dir=${data_dir} \
                 --ps_hosts=localhost:2222 \
                 --worker_hosts=localhost:2223,localhost:2224,localhost:2225 \
                 --job_name=worker --task_index=0 &
sleep 1
python DeepFM.py --learning_rate=0.0005 --optimizer=Adam --num_epochs=1 \
                 --batch_size=256 --field_size=39 --feature_size=117581 \
                 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 \
                 --num_threads=8 --model_dir=${model_dir}/DeepFM/ --data_dir=${data_dir} \
                 --ps_hosts=localhost:2222 \
                 --worker_hosts=localhost:2223,localhost:2224,localhost:2225 \
                 --job_name=worker --task_index=1 &
sleep 1
python DeepFM.py --learning_rate=0.0005 --optimizer=Adam --num_epochs=1 \
                 --batch_size=256 --field_size=39 --feature_size=117581 \
                 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 \
                 --num_threads=8 --model_dir=${model_dir}/DeepFM/ --data_dir=${data_dir} \
                 --ps_hosts=localhost:2222 \
                 --worker_hosts=localhost:2223,localhost:2224,localhost:2225 \
                 --job_name=worker --task_index=2 &
