#!/bin/bash
#SBATCH --job-name=res34res18_lr1e-3_wd1e-3_stage25_1024_mix_w07_newinit_bt512
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res34res18_lr1e-3_wd1e-3_stage25_1024_mix_w07_newinit_bt512.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -p dvlab
#SBATCH -x proj195,proj194,proj199

source activate py3.8_pt1.8.1
python main_distill_stage25.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8888' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 1e-3 \
	       --weight-decay 1e-3 \
	       --teacher_arch resnet34 \
	       --smooth 0.7 \
	       --diffusion_batch_mul 50 \
	       --diffloss_w 1024 \
	       --warmup_epochs 0 \
	       -j 32 \
	       -b 512 \
	       --mark 'workdir/res34res18_lr1e-3_wd1e-3_stage25_1024_mix_w07_newinit_bt512' \
	       --reload workdir/res34res18_lr1e-3_wd1e-3_stage1/model_best.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet
