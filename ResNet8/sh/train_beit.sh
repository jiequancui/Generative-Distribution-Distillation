#!/bin/bash
#SBATCH --job-name=beit_lr2e-3_wd2e-2_w09_mix_1024_300e_randaug_resume
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=beit_lr2e-3_wd2e-2_w09_mix_1024_300e_randaug_resume.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj195

source activate py3.8_pt1.8.1
python main_distill_beit.py -a resnet50 \
	       --dist-url 'tcp://127.0.0.1:8887' \
               --dist-backend 'nccl' \
	       --multiprocessing-distributed \
	       --world-size 1 \
	       --rank 0 \
	       --lr 2e-3 \
	       --weight-decay 2e-2 \
	       --teacher_arch beitv2_base_patch16_224 \
	       --teacher_model /mnt/proj205/jqcui/code/imagenet_cls/vanillaKD/pretrained_models/beitv2_base_patch16_224_pt1k_ft21kto1k_new.pth \
	       --smooth 0.9 \
	       --beta 1.0 \
	       --diffusion_batch_mul 5 \
	       --diffloss_w 1024 \
	       --warmup_epochs 10 \
	       --epochs 300 \
	       -j 64 \
	       -b 512 \
	       --aug_type 'randaug' \
	       --mark 'workdir/beit_lr2e-3_wd2e-2_w09_mix_1024_300e_randaug_resume' \
	       --resume workdir/beit_lr2e-3_wd2e-2_w09_mix_1024_300e_randaug/checkpoint.pth.tar \
	       /mnt/proj198/jqcui/Data/ImageNet
