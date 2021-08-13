#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train_deep_globe.py \
--task_name "all" \
--mode 4 \
--dataset 1 \
--batch_size 6 \
--sub_batch_size 8 \
--size_p 508 \
--size_g 508 \
--pre_path all.epoch.pth \
--c_path medium.epoch.pth \
--glo_path global.epoch.pth 
