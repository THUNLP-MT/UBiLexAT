source config

reconWeight=1
THEANO_FLAGS=floatX=float32 python src/linear_ae_ff_stable.py $config $lang1 $lang2 --num-minibatches 500000 --alt-loss --recon-weight $reconWeight > $reconWeight.log

cut -d ' ' -f 1 data/$config/vocab-freq.$lang1 > data/$config/vocab.$lang1
python3 scripts/translate.py data/$config/transformed-1.$lang1 data/$config/word2vec.$lang2 data/$config/vocab.$lang1 data/$config/result.1
