split=valid
nshard=10
lab_dir=/tmp2/willymen/librispeech/LibriSpeech/dev-clean/labels

for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km