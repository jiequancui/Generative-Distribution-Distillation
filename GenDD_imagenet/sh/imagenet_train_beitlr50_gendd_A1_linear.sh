#!/bin/bash
#SBATCH --job-name=imagenet_beitlargeres50_gendd_A1_linear
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenet_beitlargeres50_gendd_A1_linear.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -p dvlab
#SBATCH -w proj192

source activate py3.8_pt1.8.1

python main_supervised_gendd_linear.py -a resnet50 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 30 \
	       --cos \
	       --weight-decay 0 \
	       --teacher_arch beitv2_large_patch16_224 \
	       --teacher_model /mnt/proj205/jqcui/code/imagenet_cls/vanillaKD/pretrained_models/beitv2_large_patch16_224_pt1k_ft21kto1k_new.pth \
	       --epochs 100 \
	       --mixup 0.2 \
	       --cutmix 1.0 \
	       --smoothing 0.1 \
               --aug_type 'rand' \
	       --crop_scale 0.6 \
	       -j 48 \
	       -b 512 \
	       --mark 'workdir/imagenet_beitlargeres50_gendd_A1_linear' \
	       --reload workdir/imagenet_beitlargeres50_gendd_A1/model_best.pth.tar \
	       --finetune_classifier \
	       --alpha 0.5 \
	       /mnt/proj198/jqcui/Data/ImageNet
