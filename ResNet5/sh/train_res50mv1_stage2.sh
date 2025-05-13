#!/bin/bash
#SBATCH --job-name=res50mv1_lr1e-3_wd5e-2_stage2_1024_mix
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res50mv1_lr1e-3_wd5e-2_stage2_1024_mix.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -x proj194,proj193

source activate py3.8_pt1.8.1
python main_distill_stage2.py -a MobileNetV1 \
	       --dist-url 'tcp://127.0.0.1:8888' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 1e-3 \
	       --weight-decay 5e-2 \
	       --teacher_arch resnet50 \
	       --mark 'workdir/res50mv1_lr1e-3_wd5e-2_stage2_1024_mix' \
	       --reload 'workdir/res50mv1_lr1e-3_wd5e-2_stage1/model_best.pth.tar' \
	       /mnt/proj198/jqcui/Data/ImageNet
