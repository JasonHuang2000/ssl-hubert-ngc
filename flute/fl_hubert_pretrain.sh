# CUDA_VISIBLE_DEVICES=0,1,4,6 \
python -m torch.distributed.run \
--nproc_per_node=4 \
--master_port=25641 e2e_trainer.py \
-dataPath ./testing -outputPath scratch \
-config ./experiments/hubert_pretrain/hubert_pretrain_config.yaml \
-task mlm_bert \
-backend nccl