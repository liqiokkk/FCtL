export CUDA_VISIBLE_DEVICES=0
python train_deep_globe.py \
--task_name "all" \
--mode 3 \
--dataset 2 \
--batch_size 6 \
--sub_batch_size 16 \
--size_p 508 \
--size_g 508 \
--glo_path_10 "B10.epoch.pth" \
--glo_path_15 "B15.epoch.pth" \
--pre_path "all.epoch.pth"

