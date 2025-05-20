#!/bin/bash
#SBATCH --job-name=res50mv1_lr2e-3_wd2e-2_stage22_2048_mix_w07_bt2048_coswarm_mul5
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res50mv1_lr2e-3_wd2e-2_stage22_2048_mix_w07_bt2048_coswarm_mul5.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj202

source activate py3.8_pt1.8.1
python main_distill_stage22_cosine.py -a MobileNetV1 \
	       --dist-url 'tcp://127.0.0.1:8886' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 5e-5 \
	       --weight-decay 5e-3 \
	       --teacher_arch resnet50 \
	       --adam_beta 0.999 \
	       --smooth 0.7 \
	       --beta 1.0 \
	       --diffusion_batch_mul 5 \
	       --diffloss_w 2048 \
	       --warmup_epochs 10 \
	       --epochs 100 \
	       -j 64 \
	       -b 2048 \
	       --mark 'workdir/tmp' \
	       /mnt/proj198/jqcui/Data/ImageNet


#--reload workdir/res50mv1_lr1e-3_wd5e-2_stage1/model_best.pth.tar \

