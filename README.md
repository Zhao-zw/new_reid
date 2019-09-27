the data should put in new_reid/tri_loss/dataset

PCB train :python script/experiment/train_PCB.py \
-d '(0,1)' \
--only_test false \
--dataset market1501 \
--last_conv_stride 1 \
--normalize_feature false \
--trainset_part train \
--exp_dir Result/PCB \
--steps_per_log 10 \
--epochs_per_val 50

PCB test :python script/experiment/train_PCB.py \
-d '(0,)' \
--only_test true \
--dataset market1501 \
--last_conv_stride 1 \
--normalize_feature false \
--exp_dir Result/PCB \
--model_weight_file new_reid/Result/PCB/ckpt.pth

triploss baseline :python script/experiment/train.py \
-d '(0,1)' \
--only_test false \
--dataset market1501 \
--last_conv_stride 1 \
--normalize_feature false \
--trainset_part train \
--exp_dir Result/baseline \
--steps_per_log 10 \
--epochs_per_val 50

triploss baseline test:python script/experiment/train.py \
-d '(0,)' \
--only_test true \
--dataset market1501 \
--last_conv_stride 1 \
--normalize_feature false \
--exp_dir new_train \
--model_weight_file new_reid/Result/baseline/ckpt.pth

triploss with tripgan train :python script/experiment/train_TG_R.py \
-d '(0,1)' \
--only_test false \
--resume true \
--dataset market1501 \
--last_conv_stride 1 \
--normalize_feature false \
--trainset_part train \
--exp_dir Result/tlg/R \
--steps_per_log 1 \
--epochs_per_val 50 \
--use_FDGAN_module true \
