CUDA_VISIBLE_DEVICES=1 python fairseq_cli/hydra_train.py \
  --config-dir /tmp2/willymen/fairseq/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/tmp2/willymen/librispeech/LibriSpeech/dev-clean/manifest \
  task.label_dir=/tmp2/willymen/librispeech/LibriSpeech/dev-clean/labels \
  task.labels='["km"]' model.label_rate=100 \