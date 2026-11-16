# CAI 4105 Challenge 2 — Online Purchase Prediction

**Team 4 · Dariya, Bakai, Niyaz, and Rimma**

## Project Goal

Predict whether an online shopping session will result in a purchase
(`Revenue=True`). The training data contains 9,247 sessions, and 15.48%
resulted in a purchase.

## Data and Setup

1. Download `challenge2_train.csv` and `challenge2_test.csv` from the
   Challenge 2 assignment in Canvas.
2. Place both files in the project folder alongside
   `01_data_preprocessing.ipynb`.
3. Install the dependencies from the project folder:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Open `01_data_preprocessing.ipynb` in JupyterLab and run the cells
   from top to bottom.

## Approach

### Preprocessing
We use `Revenue` as the target and exclude `ID` from the predictors.
The preprocessing pipeline standardizes numerical features and one-hot
encodes categorical features. It is fitted on the training portion
of each fold (no leakage into validation data). Select models use 
model-specific preprocessing rather than the shared preprocessing 
(Bernoulli NB uses binned numeric features, and Gaussian NB uses 
Yeo-Johnson transformed numeric features with `ExitRates` dropped).

### Baselines, then tuning
Required models are first evaluated with default or simple settings.
We then use `GridSearchCV` to tune select models.

### Validation
We compare models using the same five stratified group folds. Sessions
with identical feature values stay in the same fold. The primary metric
is F1 for the positive class (`Revenue=True`). We use the test dataset
only to generate final predictions.

## Models Evaluated

All seven required model families were evaluated with baseline settings and
selectively tuned with Grid Search.

| Model | Baseline setup | Grid Search tuned over |
| --- | --- | --- |
| Logistic Regression | Default, no class weights | Penalty (L1/L2), `C`, `class_weight` |
| Naïve Bayes (Bernoulli, Gaussian) | Default settings, model-specific preprocessing | `alpha` / `var_smoothing`, class priors |
| K-Nearest Neighbors | `K=5`, uniform voting, Euclidean | `K`, weighting, distance metric |
| Support Vector Machine | Linear and RBF, `C=1`, balanced class weights | Kernel, `C`, `gamma`, `class_weight` |
| Decision Tree | Defaults | N/A |
| Random Forest | 300 trees | `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight` |
| Gradient Boosting | Defaults | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |

## Team Responsibilities

| Team member | Contribution |
| --- | --- |
| Dariya | Data checks, preprocessing, shared validation splits, and environment setup |
| Rimma | Logistic Regression, Naïve Bayes, KNN, SVM, and the model comparison table |
| Bakai | Decision Tree, Random Forest, Gradient Boosting, and GridSearchCV |
| Niyaz | Final notebook, confusion matrix, final model training, and `predictions.csv` |

## Model Comparison and Final Results

The team will add the validation results for all seven models, the
reason for selecting the final model, and its positive-class F1 score.

The final `predictions.csv` will contain `ID` and `Revenue` for all
3,083 test sessions, with predictions recorded as `True` or `False`.
