#!/bin/bash
#SBATCH --job-name=res50mv1_lr5e-3_wd2e-2_stage1
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=res50mv1_lr5e-3_wd2e-2_stage1.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -x proj197

source activate py3.8_pt1.8.1
python main_distill_stage1.py -a MobileNetV1 \
	       --dist-url 'tcp://127.0.0.1:8888' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 5e-3 \
	       --weight-decay 2e-2 \
	       --teacher_arch resnet50 \
	       --mark 'workdir/res50mv1_lr5e-3_wd2e-2_stage1' \
	       /mnt/proj198/jqcui/Data/ImageNet
