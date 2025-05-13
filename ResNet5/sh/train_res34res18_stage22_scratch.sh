#!/bin/bash
#SBATCH --job-name=scatch_res34res18_lr1e-3_wd1e-3_stage22_1024_mix_w10
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=scratch_res34res18_lr1e-3_wd1e-3_stage22_1024_mix_w10.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -p dvlab
#SBATCH -w proj202

source activate py3.8_pt1.8.1
python main_distill_stage22.py -a resnet18 \
	       --dist-url 'tcp://127.0.0.1:8888' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 1e-3 \
	       --weight-decay 1e-3 \
	       --teacher_arch resnet34 \
	       --smooth 1.0 \
	       --diffusion_batch_mul 50 \
	       --diffloss_w 1024 \
	       --warmup_epochs 0 \
	       -j 32 \
	       --mark 'workdir/scratch_res34res18_lr1e-3_wd1e-3_stage22_1024_mix_w10' \
	       /mnt/proj198/jqcui/Data/ImageNet
