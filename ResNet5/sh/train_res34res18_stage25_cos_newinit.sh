#!/bin/bash
#SBATCH --job-name=res34res18_lr1e-3_wd5e-2_stage25_1024_mix_w07_bt512_cos_new4_minlr0
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res34res18_lr1e-3_wd5e-2_stage25_1024_mix_w07_bt512_cos_new4_minlr0.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj193

source activate py3.8_pt1.8.1
python main_distill_stage25_cosine.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 1e-3 \
	       --weight-decay 5e-2 \
	       --teacher_arch resnet34 \
	       --teacher_model ../ResNet_baseline/workdir/resnet34_baseline/model_best.pth.tar \
	       --smooth 0.7 \
	       --beta 1.0 \
	       --diffusion_batch_mul 50 \
	       --diffloss_w 1024 \
	       --warmup_epochs 10 \
	       --epochs 100 \
	       -j 64 \
	       -b 512 \
	       --mark 'workdir/tmp' \
	       --evaluate \
	       /mnt/proj198/jqcui/Data/ImageNet


#--reload workdir/res34res18_lr1e-3_wd5e-2_stage1/model_best.pth.tar \


