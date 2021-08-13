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

#train 训练 -- not train 测试
#val 验证
#pre 预有模型
#                   train   val   pre 
#训练模型(不验证)       1      0     0
#训练模型(验证)         1      1     0
#继续训练模型(不验证)    1      0     1      
#继续训练模型(验证)     1      1      1
#测试模型              0      0     1

# export CUDA_VISIBLE_DEVICES=0
# python train_deep_globe.py \
# --task_name "global" \
# --mode 0 \
# --dataset 1 \
# --batch_size 4 \
# --sub_batch_size 1 \
# --size_p 508 \
# --size_g 508 \
# --train \
# --val 

# export CUDA_VISIBLE_DEVICES=0
# python train_deep_globe.py \
# --task_name "fcn" \
# --mode 1 \
# --dataset 1 \
# --batch_size 2 \
# --sub_batch_size 2 \
# --size_p 508 \
# --size_g 508 \
# --train \
# --val 

# export CUDA_VISIBLE_DEVICES=0
# python train_deep_globe.py \
# --task_name "dnl" \
# --mode 2 \
# --dataset 1 \
# --batch_size 2 \
# --sub_batch_size 2 \
# --size_p 508 \
# --size_g 508 \
# --train \
# --val \
# --glo_path "global.epoch49.pth" \
# --pre \
# --pre_path "fcn.epoch49.pth"


