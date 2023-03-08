n_shards=10

for ((i = 0 ; i < $n_shards ; i++)); do
    python dump_mfcc_feature.py /tmp2/willymen/librispeech/LibriSpeech/dev-clean/manifest valid $n_shards $i /tmp2/willymen/librispeech/LibriSpeech/dev-clean/features
done