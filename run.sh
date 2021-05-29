#!/bin/bash/
N=$1

declare -a datalist=("vowels" "pima" "optdigits" "letter" "cardio" "arrhythmia" "breastw" "musk" "mnist" "satimage-2" "satellite" "mammography" "thyroid" "annthyroid" "ionosphere" "pendigits" "shuttle")

for data in "${datalist[@]}"
  do
    for missing_ratio in 0.0
      do
          for oe in 0.0
              do
                  for seed in 0 1 2 3 4
                      do
                         python3 trainRCA.py  --oe $oe --batch_size 128 --data $data  --seed $seed --missing_ratio $missing_ratio --training_ratio 0.599 --validation_ratio 0.001 --max_epochs 200 --hidden_dim 128 --z_dim 10&
                        python3 train_IF.py  --data $data --seed $seed --missing_ratio $missing_ratio --training_ratio 0.599 --validation_ratio 0.001 &
                        python3 train_oneclasssvm.py  --data $data --missing_ratio $missing_ratio --seed $seed --training_ratio 0.599 --validation_ratio 0.001&
                        python3 train_DAGMM.py  --data $data --seed $seed --batch_size 128 --missing_ratio $missing_ratio --training_ratio 0.6 --z_dim 1 --max_epochs 100&
                        python3 train_SOGAAL.py  --data $data --seed $seed --missing_ratio $missing_ratio --training_ratio 0.599 --validation_ratio 0.001 --batch_size 128 --max_epochs 20 --z_dim 10&
                        python3 trainSVDD.py  --data $data --batch_size 128 --seed $seed --missing_ratio $missing_ratio --training_ratio 0.599 --validation_ratio 0.001  --max_epochs 100  --hidden_dim 128 --z_dim 10&
                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                      done
              done
      done
  done




