n_clusters=100
lab_dir=/tmp2/willymen/librispeech/LibriSpeech/dev-clean/labels

for x in $(seq 0 $((n_clusters - 1))); do
    echo "$x 1"
done >> $lab_dir/dict.km.txt