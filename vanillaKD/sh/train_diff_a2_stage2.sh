#!/bin/bash
#SBATCH --job-name=beit_r50
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=beit_r50.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj201



source activate py3.8_pt1.8.1
python -m torch.distributed.launch --nproc_per_node=8 train_diff_stage2.py /mnt/proj198/jqcui/Data/ImageNet \
       --model resnet50 \
       --teacher beitv2_base_patch16_224 \
       --teacher-pretrained pretrained_models/beitv2_base_patch16_224_pt1k_ft21kto1k_new.pth \
       --kd-loss kd \
       --amp \
       --epochs 300 \
       --batch-size 256 \
       --lr 2e-3 \
       --opt lamb \
       --sched cosine \
       --weight-decay 0.02 \
       --warmup-epochs 20 \
       --warmup-lr 1e-6 \
       --smoothing 0.0 \
       --drop 0 \
       --drop-path 0.05 \
       --aug-repeats 3 \
       --aa rand-m7-mstd0.5 \
       --mixup 0.1 \
       --cutmix 1.0 \
       --color-jitter 0 \
       --crop-pct 0.95 
