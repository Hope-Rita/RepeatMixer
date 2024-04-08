# Codes for "Repeat-Aware Neighbor Sampling for Dynamic Graph Learning"


Datasets are from [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o), The preprocess for datasets is the same as [Towards Better Dynamic Graph Learning: New Architecture and Unified Library](https://github.com/yule-BUAA/DyGLib).


### Model Training

Example of training RepeatMixer on Wikipedia dataset:

```python train_link_prediction.py --dataset_name wikipedia --model_name RepeatMixer  --num_runs 5 --gpu 0 --load_best_configs```

### Model Evaluation

Three (i.e., random, historical, and inductive) negative sampling strategies can be used for model evaluation.

```python evaluate_link_prediction.py --dataset_name wikipedia --model_name RepeatMixer --negative_sample_strategy random --num_runs 5 --gpu 0```
