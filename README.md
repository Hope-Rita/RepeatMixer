# Codes for "Repeat-Aware Neighbor Sampling for Dynamic Graph Learning"


Datasets are from [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o), The preprocess for datasets is the same as [Towards Better Dynamic Graph Learning: New Architecture and Unified Library](https://github.com/yule-BUAA/DyGLib).


### Model Training

Example of training RepeatMixer on Wikipedia dataset:

```python train_link_prediction.py --dataset_name wikipedia --model_name RepeatMixer  --num_runs 5 --gpu 0 --load_best_configs```

### Model Evaluation

Three (i.e., random, historical, and inductive) negative sampling strategies can be used for model evaluation.

```python evaluate_link_prediction.py --dataset_name wikipedia --model_name RepeatMixer --negative_sample_strategy random --num_runs 5 --gpu 0```

### Performance compared to FreeDyG

| Datasets  | FreeDyG         | RepeatMixer(F)  | RepeatMixer |
| --------- | --------------- | --------------- | ----------- |
| Wikipedia | 98.92 ± 0.02 | 99.00 ± 0.03 | **99.16** ± 0.02 |
| Reddit    | **99.25** ± 0.01 | 99.14 ± 0.01 | 99.22 ± 0.01|
| MOOC      | 87.76 ± 0.68 | 84.95 ± 0.26 | **92.76** ± 0.10 |
| LastFM    | 92.15 ± 0.16 | 91.44 ± 0.05 | **94.14** ± 0.06 |
| Enron     | 91.15 ± 0.20   | 92.07 ± 0.07   | **92.66** ± 0.07 |
| UCI       | 95.85 ± 0.14   | 96.33 ± 0.14  | **96.74** ± 0.08 |

### Similarity based on features
We calculate the similarity based on edge features on Wikipedia, Reddit, and MOOC since only these three datasets contain edge features. Specifically, we calculate the cosine similarity between node pairs based on their neighbors' edge feature sequences. We calculate the average value of all positive and negative pairs in the datasets.

|      | recent NSS | Wikipedia | Reddit | MOOC |
| ---- | ---------- | --------- | ------ | ---- |
| pos  | train      | 15.73     | 16.83 |34.75|
|      | valid      | 18.30     | 16.70 |40.82|
|      | test       | 18.30     | 16.70 |40.82|
| neg  | train      | 6.40      | 2.94 |12.80|
|      | valid      | 4.16      | 2.64 |7.74|
|      | test       | 4.14      | 2.69 |7.57|

|      | repeat-aware NSS | Wikipedia | Reddit | MOOC |
| ---- | ---------- | --------- | ------ | ---- |
| pos  | train      | 33.49 | 45.07  |66.95|
|      | valid      | 34.02 |46.16|67.74|
|      | test       | 34.02 |46.16|67.74|
| neg  | train      | 6.70   |3.39|29.11|
|      | valid      | 4.26   |2.95|12.00|
|      | test       | 4.28     |2.99|12.03|

### Similarity based on structural information
We set the value of similarity based on structural information as 1 if they have common neighbors or have repeat behaviors, otherwise is set as 0. We calculate the average value of all positive and negative pairs in the datasets.

||recent NSS|Wikipedia|LastFM|UCI| enron |
|-|-|-|-|-| ----- |
|pos|train|86.95|69.42|79.65|95.98 |
||valid|84.03|73.40|71.11|93.62 |
||test |84.03|73.51|73.33|93.62
|neg|train|0.25|7.19|14.53|11.49 
||valid|0.0|7.52|6.66| 8.51 
||test|0.84|8.76|2.22|5.32 

||repeat-aware NSS|Wikipedia|LastFM|UCI|Enron |
|-|-|-|-|-| ----- |
|pos|train|87.19|86.28|81.39|98.27
||valid|84.03|82.06|71.11|97.87 
||test|84.03|82.06|73.33|97.87 
|neg|train|1.23|17.70|11.05|15.23 
||valid|0.0|12.78|6.66|10.64 
||test|0.84|14.64|2.22| 9.57 

### Performance in random, historical, and inductive settings
Due to the time limitations, we show the results from three settings in Table 2. The rest results will be added in the Appendix.


|            |                 | Sampling STR | Wikipedia       | Reddit          | MOOC            | LastFM          | Enron           | UCI            |
| ---------- | --------------- | ----------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| rnd        | transductive    | uniform NSS          | 0.9475 ± 0.0073 | 0.9753 ± 0.0015 | 0.7291 ± 0.0102 | 0.6715 ± 0.0032 | 0.7022 ± 0.0208 | 0.8902 ± 0.0060 |
|            |            | time-aware NSS| 0.9455 ± 0.0067 | 0.7291 ± 0.0102   | 0.7288 ± 0.0081 | 0.6647 ± 0.0013 | 0.7074 ± 0.0156 | 0.8887 ± 0.0073 |
|            |            | recent  NSS   | 0.9766 ± 0.0024 | 0.9796 ± 0.0022   | 0.8393 ± 0.0019 | 0.8224 ± 0.0021 | 0.8461 ± 0.0178 | 0.9451 ± 0.0016 |
|            |            | repeat-aware NSS       | **0.9900** ± 0.0003 | **0.9914** ± 0.0001 | **0.8495** ± 0.0026 | **0.9144** ± 0.0005 | **0.9207** ± 0.0007 | **0.9633** ± 0.0014 |
|            | inductive  | uniform         | 0.9393 ± 0.0076   | 0.9458 ± 0.0045 | 0.7493 ± 0.0104 | 0.7639 ± 0.0024 | 0.5654 ± 0.0077 | 0.8388 ± 0.0073 |
|            |            | time-aware | 0.9377 ± 0.0067 | 0.7493 ± 0.0104   | 0.7513 ± 0.0092 | 0.7601 ± 0.0016 | 0.5681 ± 0.0204 | 0.8405 ± 0.0110 |
|            |            | recent     | 0.9720 ± 0.0022 | 0.9635 ± 0.0051   | 0.8277 ± 0.0023 | 0.8660 ± 0.0033 | 0.7784 ± 0.0199 | 0.9223 ± 0.0020 |
|            |            | Ours       | **0.9862** ± 0.0003 | **0.9868** ± 0.0003 | **0.8454** ± 0.0031 | **0.9295** ± 0.0012 | **0.8816** ± 0.0022 | **0.9476** ± 0.0008 |


||                 | Sampling strategy | wikipedia       | reddit          | mooc            | lastfm          | enron           | uci  |
| ------------ | --------------- | ----------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ---- |
|  hist         | transductive | uniform         | 0.9093 ± 0.0025   | 0.7899 ± 0.0217 | 0.6912 ± 0.0495 | 0.8546 ± 0.0351 | 0.8694 ± 0.0029 | 0.8569 ± 0.0178 |
||| time-aware   | 0.7838 ± 0.0534 | 0.7534 ± 0.0047   | 0.6612 ± 0.0358 | 0.6790 ± 0.0200 | 0.8359 ± 0.0341 | 0.8588 ± 0.0233 |
||| recent       | 0.7803 ± 0.0230 | 0.8139 ± 0.0075   | 0.8072 ± 0.0141 | 0.7490 ± 0.0056 | 0.7585 ± 0.0406 | 0.7207 ± 0.0622 |
||| Ours         | **0.9102** ± 0.0059 | **0.8444** ± 0.0050 | **0.9424** ± 0.0025 | **0.8841** ± 0.0007 | **0.8819** ± 0.0023 | **0.8641** ± 0.0079 |
|| inductive    | uniform         | 0.8492 ± 0.0066   | 0.6346 ± 0.0050 | 0.6813 ± 0.0190 | **0.8267** ± 0.0224 | 0.8358 ± 0.0058 | 0.8242 ± 0.0155 |
||| time-aware   | 0.6737 ± 0.0649 | 0.6032 ± 0.0019   | 0.6637 ± 0.0195 | 0.7066 ± 0.0028 | 0.7470 ± 0.0616 | 0.8494 ± 0.0220 |
||| recent       | 0.7230 ± 0.0140 | 0.6661 ± 0.0153   | 0.7530 ± 0.0054 | 0.7188 ± 0.0206 | 0.7075 ± 0.0384 | 0.7154 ± 0.0367 |
||| Ours         | **0.8554** ± 0.0114 | **0.6745** ± 0.0137 | **0.8367** ± 0.0056 | 0.8123 ± 0.0002 | **0.8449** ± 0.0043 | **0.8549** ± 0.0020 |


|            |                 | Sampling STR | Wikipedia       | Reddit          | MOOC            | LastFM          | Enron           | UCI            |
| ------------ | --------------- | ----------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ---- |
| ind          | transductive | uniform  NSS       | 0.8668 ± 0.0070   | 0.8555 ± 0.0012 | 0.6701 ± 0.0497 | **0.8509** ± 0.0620 | 0.8032 ± 0.0085 | 0.7969 ± 0.0181 |
||| time-aware NSS  | 0.7126 ± 0.0355 | 0.8538 ± 0.0085   | 0.6485 ± 0.0148 | 0.6138 ±  0.0113 | 0.7544 ± 0.0378 | 0.7624 ± 0.0468 |
||| recent NSS       | 0.8875 ± 0.0080 | 0.8355 ± 0.0305   | 0.7522 ± 0.0049 | 0.6492 ± 0.0116 | 0.7404 ± 0.0263 | 0.6917 ± 0.0365 |
||| repeat-aware NSS    | **0.8898** ± 0.0149 | **0.9176** ± 0.0056 | **0.8368** ± 0.0072 | 0.7634 ± 0.0030 | **0.8417** ± 0.0024 | **0.8496** ± 0.0052 |
|| inductive    | uniform  NSS       | 0.8494 ± 0.0066   | 0.6020 ± 0.0059 | 0.6556 ± 0.0442 | **0.8267** ± 0.0223 | 0.8359 ± 0.0057 | 0.8244 ± 0.0153 |
||| time-aware NSS  | 0.6740 ± 0.0651 | 0.6032 ± 0.0019   | 0.6931 ± 0.0189 | 0.7066 ±  0.0028 | 0.7476 ± 0.0614 | 0.8492 ± 0.0216 |
||| recent  NSS     | 0.8382 ± 0.0178 | 0.6660 ± 0.0153   | 0.7529 ± 0.0054 | 0.7188 ± 0.0206 | 0.7074 ± 0.0385 | 0.7148 ± 0.0373 |
||| repeat-aware NSS    | **0.8554** ± 0.0113 | **0.6743** ± 0.0136 | **0.8367** ± 0.0056 | 0.8123 ± 0.0002 | **0.8448** ± 0.0043 | **0.8548** ± 0.0020 |
