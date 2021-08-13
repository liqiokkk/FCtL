export CUDA_VISIBLE_DEVICES=0
python train_deep_globe.py \
--task_name "global" \
--mode 0 \
--dataset 1 \
--batch_size 6 \
--sub_batch_size 6 \
--size_p 508 \
--size_g 508 \
--train \
--val 
