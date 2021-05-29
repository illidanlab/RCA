# RCA: A Deep Collaborative Autoencoder Approach for Anomaly Detection
This is the official implementation of Robust Collaborative Autoencoders(RCA). 

## Paper Abstract
Unsupervised anomaly detection plays a crucial role in many critical applications.
Driven by the success of deep learning, recent years have witnessed a growing interest in applying deep neural networks (DNNs) to anomaly detection problems. A common approach is using autoencoders to learn a feature representation for the normal observations in the data. The reconstruction error of the autoencoder is then used as outlier score to detect the anomalies. However, due to the high complexity brought upon by the over-parameterization of DNNs, the reconstruction error of the anomalies could also be small, which hampers the effectiveness of these methods. To alleviate this problem, we propose a robust framework using collaborative autoencoders to jointly identify normal observations from the data while learning its feature representation. We investigate the theoretical properties of the framework and empirically show its outstanding performance as compared to other DNN-based methods. Our experimental results also show the resiliency of the framework to missing values compared to other baseline methods.

## RCA vs Autoencoder
1. RCA uses multiple autoencoders (we found two autoencoders are usually enough).
2. For each minibatch, RCA only uses samples with small reconstruction loss to update while AE uses all samples in minibatch to update the model.
3. Each autoencoder of RCA will exchange the selected data to other autoencoder.
4. RCA still use dropout during evaluation to get multiple anomaly scores while autoencoder only uses dropout in training.

## Conda Environment
We provide the conda virtual environment in environment.yml. 

## Data
We use the ODDs dataset. The preprocessed data is in the data folder and you need first unzip the data.rar file.
More details can be found in [Official Page of ODDs Dataset](http://odds.cs.stonybrook.edu/)

## Example: 
run RCA on vowels:

> python3 trainRCA.py --data vowels --missing_ratio 0

run RCA on pima:

> python3 trainRCA.py --data pima --missing_ratio 0

run RCA on vowels with 10% missing value and mean imputation:

> python3 trainRCA.py --data vowels --missing_ratio 0.1

run RCA by using k autoencoders

> python3 trainRCAMulti.py --data vowels --missing_ratio 0.0 --n_member k

## Hyperparameters
Since in unsupervised anomaly detection, there is no clean validation data available to tune the hyperparameter. Thus, we use the same hyperparameter across all different datasets to show that our method does not heavily depend on hyperparameter tuning. 
> batchsize=128
>
>learningrate=3e-4 with Adam Optimizer
>
>hidden dimension=256
>
>bottleneck dimension=10
>
>The network structure is in the models/RCA.py. Currently, we use a 6-layer autoencoder.


## Baselines
We implement several baselines. Our implementations for one class SVM, SO-GAAL, isolation forest are based on the [pyod](https://github.com/yzhao062/pyod) implementation.
They also provide the official benchmark on ODDs dataset, which can be found in [here](https://pyod.readthedocs.io/en/latest/benchmark.html).

We implement the [DAGMM](https://openreview.net/forum?id=BJJLHbb0-) and [Deep one class SVM](http://proceedings.mlr.press/v80/ruff18a.html) by ourselves. Our DAGMM implementation heavily depends on this [third-party implementation](https://github.com/danieltan07/dagmm), and we found the DAGMM is highly numerical unstable in the ODDs dataset.
For the DeepSVDD, we train the autoencoder for 50 epochs as the initialization. 

## Acknowledgements
This research is funded by NSF-IIS 2006633, EF1638679, NSF-IIS-1749940, Office of Naval Research N00014-20-1-2382, National Institue on Aging RF1AG072449.




