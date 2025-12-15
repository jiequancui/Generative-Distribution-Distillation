#!/bin/bash
#SBATCH --job-name=imagenet_beitlargeres50_gendd_A2_linear
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imagenet_beitlargeres50_gendd_A2_linear.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -p dvlab
#SBATCH -w proj198

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
	       --mixup 0.1 \
	       --cutmix 1.0 \
	       --smoothing 0.0 \
               --aug_type 'rand' \
	       -j 32 \
	       -b 512 \
	       --reload 'workdir/imagenet_beitlargeres50_gendd_A2/model_best.pth.tar' \
	       --mark 'workdir/imagenet_beitlargeres50_gendd_A2_linear' \
	       --finetune_classifier \
	       --alpha 0.5 \
	       /mnt/proj198/jqcui/Data/ImageNet
