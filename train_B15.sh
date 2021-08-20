export CUDA_VISIBLE_DEVICES=0
python train_deep_globe.py \
--task_name "B15" \
--mode 2 \
--dataset 2 \
--batch_size 6 \
--sub_batch_size 4 \
--size_p 508 \
--size_g 508 \
--context10 3 \
--pre_path Bfcn.epoch.pth \
--train \
--val 
