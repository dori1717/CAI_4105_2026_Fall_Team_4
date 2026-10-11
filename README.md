# CAI 4105 - Fall 2026 - Team 4

## Team Members
- Dariya
- Bakai
- Rimma
- Niyaz

## Project Overview
This repository contains the work for our CAI 4105 Fall 2026 team project: predicting the monthly rental price of an apartment (`price`) from its characteristics and location. RMSE is the primary evaluation metric.

## Repository Structure

### data/
Contains the training and test datasets:
- challenge1_train.csv
- challenge1_test.csv

### notebooks/
Contains Jupyter notebooks for data preprocessing, model development, evaluation, and analysis:
- 01_data_preprocessing.ipynb: data checks, shared train/validation split, preprocessing, Linear Regression
- 02_decision_tree.ipynb: Decision Tree with Grid Search
- 03_random_forest.ipynb: Random Forest with Grid Search and feature engineering
- 04_gradient_boosting.ipynb: Gradient Boosting with Grid Search and a log-transformed target
- 05_model_comparison.ipynb: evaluation and comparison of all models under one protocol

### results/
- model_comparison.csv: all models with the same metrics (written by `05_model_comparison.ipynb`)
- decision_tree.csv, linear_regression_results.csv, gradient_boosting.csv: per-model results from the model notebooks

### requirements.txt
Pinned Python packages needed to run the notebooks.

## Validation Strategy
- Hold-out split: 80% training (5,918 rows), 20% validation (1,480 rows), `train_test_split(test_size=0.2, random_state=42)`.
- 5-fold cross-validation (`KFold(shuffle=True, random_state=42)`) on the training part for Grid Search and model selection. CV RMSE (mean ± std over folds) is the primary selection metric.
- The validation set is not used for tuning and serves as an independent check.
- Preprocessing is inside scikit-learn pipelines, so it is fit on training folds only.
- The test set is used only to generate the final predictions.

## Model Comparison
All models are retrained in `05_model_comparison.ipynb` with the hyperparameters selected by each model notebook, on the same split and the same CV folds. Sorted by CV RMSE (lower is better):

| Model | Train RMSE | CV RMSE | Validation RMSE | Validation R² |
|---|---|---|---|---|
| Gradient Boosting (tuned + log target) | 201.8 | 383.9 ± 14.1 | 405.9 | 0.77 |
| Gradient Boosting (tuned) | 183.1 | 396.1 ± 13.1 | 410.3 | 0.77 |
| Random Forest (tuned) | 181.9 | 412.5 ± 24.1 | 444.9 | 0.73 |
| Random Forest (tuned + FE) | 173.8 | 417.1 ± 26.6 | 450.8 | 0.72 |
| Random Forest (baseline) | 159.9 | 424.4 ± 22.4 | 456.5 | 0.71 |
| Gradient Boosting (baseline) | 405.2 | 462.8 ± 29.6 | 486.7 | 0.67 |
| Decision Tree (tuned) | 346.4 | 482.4 ± 16.5 | 503.7 | 0.65 |
| Linear Regression | 515.0 | 520.9 ± 35.7 | 550.0 | 0.58 |
| Decision Tree (baseline) | 39.4 | 559.8 ± 19.7 | 544.2 | 0.59 |
| Mean baseline | 814.0 | 813.3 ± 40.8 | 850.9 | 0.00 |

Key findings:
- Gradient Boosting with Grid Search and a log-transformed target has the lowest CV and validation RMSE. It beats every model outside its family in all 5 folds, and the bootstrap 95% CI of the validation difference excludes 0.
- Grid Search helped every tree model; for Gradient Boosting it lowered CV RMSE by 67, and training on `log(price)` lowered it by another 12.
- The engineered features in `03` give no significant gain over the tuned Random Forest.
- Linear Regression underfits (train and CV RMSE are almost equal and high). The unpruned Decision Tree overfits heavily (train 39 vs CV 560); pruning through Grid Search reduces the gap.
- The CV error of the tree ensembles still decreases with more training data, so the final model should be refit on all labeled rows.
- The 10% most expensive listings produce 58% of the squared error of the best model; predictions are pulled toward the mean at both ends of the price range.

## Final Model
Best of the required models: **Gradient Boosting, tuned, trained on log(price)** (`n_estimators=1000`, `learning_rate=0.1`, `max_depth=5`, `min_samples_leaf=5`, `subsample=0.8`, `max_features=0.5`), with CV RMSE 383.9 ± 14.1 and validation RMSE 405.9. It is the starting point for the final model selection, which adds location features and ensembles under the same rule: lowest CV RMSE wins, and when the difference is not significant the simpler and faster model is preferred.

## Reproducibility
Requires Python 3.12 or newer (results in this README were produced with Python 3.14). Exact package versions are pinned in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Notebooks are run from the `notebooks/` directory and read data from `../data/`. All random seeds are fixed (`random_state=42`).

## Workflow
Each team member works on a separate Git branch. Changes are committed and pushed to the individual branch, then merged into the main branch through a pull request.
