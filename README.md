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
| Reddit    |  | 99.14 ± 0.01 | 99.22 ± 0.01|
| MOOC      | 87.76 ± 0.68 | 84.95 ± 0.26 | 92.76 ± 0.10|
| LastFM    |  | 91.44 ± 0.05 | 94.14 ± 0.06|
| Enron     | 91.15 ± 0.20   | 92.07 ± 0.07   |  92.66 ± 0.07|
| UCI       | 95.85 ± 0.14   | 96.33 ± 0.14  | 96.74 ± 0.08|
