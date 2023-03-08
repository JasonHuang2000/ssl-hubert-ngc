n_shards=10

for ((i = 0 ; i < $n_shards ; i++)); do
    python dump_km_label.py /tmp2/willymen/librispeech/LibriSpeech/dev-clean/features valid ./k_means_model_valid $n_shards $i /tmp2/willymen/librispeech/LibriSpeech/dev-clean/labels
done
