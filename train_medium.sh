export CUDA_VISIBLE_DEVICES=0
python train_deep_globe.py \
--task_name "medium" \
--mode 2 \
--dataset 1 \
--batch_size 6 \
--sub_batch_size 4 \
--size_p 508 \
--size_g 508 \
--pre_path Afcn.epoch.pth \
--train \
--val 


