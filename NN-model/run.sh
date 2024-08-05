# Model pretrain over small scale topologies
python3 train.py --mode classify --train-data-dir MCF_BRITE_small_train --valid-data-dir MCF_BRITE_small_train --use-cuda --training-epochs 500 --log-dir pretrain_small

# pretrain for general classification model
python3 train.py --mode classify --train-data-dir MCF_train --valid-data-dir MCF_BRITE_small_train --use-cuda --training-epochs 1000 --log-dir pretrain_calssify --model-load-dir pretrain_small
# pretrain for general regression model
python3 train.py --mode normal --train-data-dir MCF_train --valid-data-dir MCF_BRITE_small_train --use-cuda --training-epochs 1000 --log-dir pretrain_normal --model-load-dir pretrain_small

# P2 training for large-scale topologies, build up a dataset for a specific large-scale topology (e.g., MCF_DialtelecomCz_test)
python3 train.py --mode classify --train-data-dir MCF_DialtelecomCz_test --valid-data-dir MCF_DialtelecomCz_test --use-cuda --training-epochs 10 --log-dir P2_DialtelecomCz  --part-failure --model-load-dir pretrain_classify --batch-size 1

# test general classification model
python3 eval.py --mode classify --eval-data-dir MCF_BRITE_small_vary_capa_test1 --use-cuda  --log-dir pretrain_classify 

# test general regression model
python3 eval.py --mode normal --eval-data-dir MCF_BRITE_small_vary_capa_test1 --use-cuda  --log-dir pretrain_normal 

# test general classification model for tripple failures
python3 eval.py --mode classify --eval-data-dir MCF_tripple_test --use-cuda  --log-dir pretrain_classify --failure-type tripple

# detect the critical failure scenarios
python3 detect_critical_failure.py  --eval-data-dir MCF_BRITE_small_vary_capa_test1 --use-cuda  --detect-classify-model-dir pretrain_classify --detect-normal-model-dir pretrain_normal