#!/bin/bash
#SBATCH --job-name=res50mv1_lr01_wd1e-4_stage26_kd_cropo008
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res50mv1_lr01_wd1e-4_stage26_kd_crop008.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj202

source activate py3.8_pt1.8.1
python main_distill_stage26_cosine_kd.py -a MobileNetV1 \
	       --dist-url 'tcp://127.0.0.1:8886' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 0.1 \
	       --weight-decay 1e-4 \
	       --crop_scale 0.08 \
	       --teacher_arch resnet50 \
	       --warmup_epochs 10 \
	       --epochs 100 \
	       -j 64 \
	       -b 512 \
               --mark 'workdir/res50mv1_lr01_wd1e-4_stage26_cos_kd_crop008' \
	       /mnt/proj198/jqcui/Data/ImageNet
