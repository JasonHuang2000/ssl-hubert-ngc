python -m torch.distributed.run \
--nproc_per_node=5 e2e_trainer.py \
-dataPath ./testing -outputPath scratch \
-config ./experiments/hubert_pretrain/hubert_pretrain_config.yaml \
-task mlm_bert \
-backend nccl