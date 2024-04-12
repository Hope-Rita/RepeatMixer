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
| Wikipedia | 98.92 ± 0.02 | 99.00 ± 0.03 | 99.16 ± 0.02|
| Reddit    | 99.25 ± 0.01 | 99.14 ± 0.01 | 99.22 ± 0.01|
| MOOC      | 87.76 ± 0.68 | 84.95 ± 0.26 | 92.76 ± 0.10|
| LastFM    | 92.15 ± 0.16 | 91.44 ± 0.05 | 94.14 ± 0.06|
| Enron     | 91.15 ± 0.20   | 92.07 ± 0.07   |  92.66 ± 0.07|
| UCI       | 95.85 ± 0.14   | 96.33 ± 0.14  | 96.74 ± 0.08|

### Similarity based on features

### Similarity based on structural information
We set the value of similarity based on structural information as 1 if they have common neighbors or have repeat behaviors, otherwise is set as 0. We calculate the average value of all positive pairs and negative pairs in the datasets.

||recent NSS|Wiki|LastFM|UCI| enron |
|-|-|-|-|-| ----- |
|pos|train|86.95|69.42|79.65|95.98 |
||valid|84.03|73.40|71.11|93.62 |
||test |84.03|73.51|73.33|93.62
|neg|train|0.25|7.19|14.53|11.49 
||valid|0.0|7.52|6.66| 8.51 
||test|0.84|8.76|2.22|5.32 

||repeat-aware NSS|Wiki|LastFM|UCI|enron |
|-|-|-|-|-| ----- |
|pos|train|87.19|86.28|81.39|98.27
||valid|84.03|82.06|71.11|97.87 
||test|84.03|82.06|73.33|97.87 
|neg|train|1.23|17.70|11.05|15.23 
||valid|0.0|12.78|6.66|10.64 
||test|0.84|14.64|2.22| 9.57 

### Performance in random, historical, and inductive settings


|            |                 | Sampling strategy | wikipedia       | reddit          | mooc            | lastfm          | enron           | uci             |
| ---------- | --------------- | ----------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| rnd        | transductive    | uniform           | 0.9475 ± 0.0073 | 0.9753 ± 0.0015 | 0.7291 ± 0.0102 | 0.6715 ± 0.0032 | 0.7022 ± 0.0208 | 0.8902 ± 0.0060 |
|            |            | time-aware | 0.9455 ± 0.0067 | 0.7291 ± 0.0102   | 0.7288 ± 0.0081 | 0.6647 ± 0.0013 | 0.7074 ± 0.0156 | 0.8887 ± 0.0073 |
|            |            | recent     | 0.9766 ± 0.0024 | 0.9796 ± 0.0022   | 0.8393 ± 0.0019 | 0.8224 ± 0.0021 | 0.8461 ± 0.0178 | 0.9451 ± 0.0016 |
|            |            | Ours       | **0.9900** ± 0.0003 | **0.9914** ± 0.0001 | **0.8495** ± 0.0026 | **0.9144** ± 0.0005 | **0.9207** ± 0.0007 | **0.9633** ± 0.0014 |
|            | inductive  | uniform         | 0.9393 ± 0.0076   | 0.9458 ± 0.0045 | 0.7493 ± 0.0104 | 0.7639 ± 0.0024 | 0.5654 ± 0.0077 | 0.8388 ± 0.0073 |
|            |            | time-aware | 0.9377 ± 0.0067 | 0.7493 ± 0.0104   | 0.7513 ± 0.0092 | 0.7601 ± 0.0016 | 0.5681 ± 0.0204 | 0.8405 ± 0.0110 |
|            |            | recent     | 0.9720 ± 0.0022 | 0.9635 ± 0.0051   | 0.8277 ± 0.0023 | 0.8660 ± 0.0033 | 0.7784 ± 0.0199 | 0.9223 ± 0.0020 |
|            |            | Ours       | **0.9862** ± 0.0003 | **0.9868** ± 0.0003 | **0.8454** ± 0.0031 | **0.9295** ± 0.0012 | **0.8816** ± 0.0022 | **0.9476** ± 0.0008 |

