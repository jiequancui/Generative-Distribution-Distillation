#!/bin/bash
#SBATCH --job-name=res50mv1_lr1e-3_wd5e-2_stage26_2048_mix_w07_bt512_cos
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res50mv1_lr1e-3_wd5e-2_stage26_2048_mix_w07_bt512_cos.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj199

source activate py3.8_pt1.8.1
python main_distill_stage26_cosine.py -a MobileNetV1 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 1e-3 \
	       --weight-decay 5e-2 \
	       --teacher_arch resnet50 \
	       --smooth 0.7 \
	       --beta 1.0 \
	       --diffusion_batch_mul 50 \
	       --diffloss_w 2048 \
	       --warmup_epochs 10 \
	       --epochs 100 \
	       -j 64 \
	       -b 512 \
	       --mark 'workdir/res50mv1_lr1e-3_wd5e-2_stage26_2048_mix_w07_bt512_cos' \
               --reload workdir/res50mv1_lr1e-3_wd5e-2_stage1/model_best.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet
