# CAI 4105 Challenge 2 — Online Purchase Prediction

**Team 4 · Dariya, Bakai, Niyaz, and Rimma**

- **Final notebook:** [04_final_model.ipynb](04_final_model.ipynb)
- **Final predictions:** [predictions.csv](predictions.csv)

## Project Goal

Predict whether an online shopping session will result in a purchase
(`Revenue=True`). The training data contains 9,247 sessions, and 15.48%
resulted in a purchase.

## Files

| File | Content |
| --- | --- |
| `04_final_model.ipynb` | Final notebook: the complete workflow from loading the data to `predictions.csv` |
| `predictions.csv` | Final predictions (`ID`, `Revenue`) for the 3,083 test sessions |
| `01_data_preprocessing.ipynb` | Data checks, shared preprocessing, and validation setup |
| `02_models_1-4.ipynb` | Logistic Regression, Naïve Bayes, KNN, and SVM with Grid Search |
| `03_bakai_models.ipynb` | Decision Tree, Random Forest, and Gradient Boosting with Grid Search |
| `challenge2_train.csv`, `challenge2_test.csv` | Training and test data |
| `requirements.txt` | Pinned Python packages |

The individual notebooks `01` to `03` show how each part was developed.
The final notebook repeats all of their steps under one shared validation
setup, so it is the only notebook that needs to be run.

## Data and Setup

The results were produced with Python 3.14.

1. `challenge2_train.csv` and `challenge2_test.csv` are in this folder,
   next to the notebooks. If they are missing, download them from the
   Challenge 2 assignment in Canvas and place them here.
2. Install the dependencies from this folder:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Start JupyterLab with `jupyter lab`, open `04_final_model.ipynb`, and
   run the cells from top to bottom. The notebook takes a few minutes
   and writes `predictions.csv` into this folder.

Random seeds are fixed (`random_state=42`), so a rerun reproduces the
same validation results and the same `predictions.csv`.

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
We then use `GridSearchCV` to tune each model on the same folds.

### Validation
We compare models using the same five stratified group folds. Sessions
with identical feature values stay in the same fold. The primary metric
is F1 for the positive class (`Revenue=True`). We use the test dataset
only to generate final predictions.

## Models Evaluated

All seven required model families were evaluated with baseline settings and
tuned with Grid Search.

| Model | Baseline setup | Grid Search tuned over |
| --- | --- | --- |
| Logistic Regression | Default, no class weights | Penalty (L1/L2), `C`, `class_weight` |
| Naïve Bayes (Bernoulli, Gaussian) | Default settings, model-specific preprocessing | `alpha` / `var_smoothing`, class priors |
| K-Nearest Neighbors | `K=5`, uniform voting, Euclidean | `K`, weighting, distance metric |
| Support Vector Machine | Linear and RBF, `C=1`, balanced class weights | Kernel, `C`, `gamma`, `class_weight` |
| Decision Tree | Defaults | `max_depth`, `min_samples_leaf`, `class_weight` |
| Random Forest | 300 trees | `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight` |
| Gradient Boosting | Defaults | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |

## Team Responsibilities

| Team member | Contribution |
| --- | --- |
| Dariya | Data checks, preprocessing, shared validation splits, and environment setup |
| Rimma | Logistic Regression, Naïve Bayes, KNN, SVM, and the model comparison table |
| Bakai | Decision Tree, Random Forest, Gradient Boosting, and GridSearchCV |
| Niyaz | Final notebook, confusion matrix, final model training, and `predictions.csv` |

## Model Comparison

The table shows the best version of each model, sorted by validation F1
for `Revenue=True` (mean ± standard deviation over the five folds). The full table with all
baseline and tuned versions is in Section 6 of the final notebook.

| Model | Best version | Validation F1 | Precision | Recall | Train F1 |
| --- | --- | --- | --- | --- | --- |
| **Random Forest** | **tuned** | **0.688 ± 0.027** | **0.682** | **0.695** | **0.944** |
| Gradient Boosting | default settings, selected by Grid Search | 0.670 ± 0.018 | 0.727 | 0.621 | 0.734 |
| Logistic Regression | tuned | 0.669 ± 0.006 | 0.627 | 0.716 | 0.669 |
| Support Vector Machine | tuned (RBF kernel) | 0.667 ± 0.012 | 0.590 | 0.767 | 0.675 |
| Decision Tree | tuned (`max_depth=4`) | 0.657 ± 0.025 | 0.727 | 0.600 | 0.674 |
| Naïve Bayes | Bernoulli, tuned | 0.614 ± 0.025 | 0.552 | 0.690 | 0.617 |
| K-Nearest Neighbors | tuned (K=3) | 0.493 ± 0.029 | 0.614 | 0.412 | 0.703 |

Main findings:

- Handling the class imbalance was the biggest improvement. Balanced class
  weights raised Logistic Regression from 0.515 to 0.669 and, together
  with larger leaves, Random Forest from 0.634 to 0.688.
- Gradient Boosting, Logistic Regression, and SVM are effectively tied
  behind the Random Forest. For Gradient Boosting, Grid Search selected
  the default settings out of 32 configurations.
- `PageValues` carries most of the signal: the tuned L1 Logistic
  Regression keeps only this feature and still reaches 0.669.
- Grid Search fixed the overfitting of the Decision Tree: limiting its
  depth and leaf size raised it from 0.572 to 0.657. KNN still overfits,
  and Naïve Bayes underfits.

## Final Model

**Tuned Random Forest** in the shared preprocessing pipeline
(`n_estimators=400`, `min_samples_leaf=2`, `max_features="sqrt"`,
`class_weight="balanced_subsample"`, no depth limit).

**Final validation performance: F1 = 0.688 for `Revenue=True`**
(precision 0.682, recall 0.695, standard deviation over folds 0.027).
On the validation predictions the model finds 994 of the 1,431 purchases,
misses 437, and raises 463 false alarms.

Why we selected it:

- **Validation performance:** it has the highest validation F1, about 0.02
  above the next three models. It beats Gradient Boosting in all five
  folds, SVM in four, and Logistic Regression in three. The margin is
  similar to the variation between folds, so the advantage is moderate.
  Its average precision (0.743) and ROC AUC (0.930) are level with
  Gradient Boosting and clearly above Logistic Regression and SVM.
- **Overfitting:** the training F1 (0.944) is well above the validation
  F1, which is expected for a forest of deep trees. Tuning reduced this
  gap from 0.366 to 0.255 while the validation F1 increased.
- **Complexity and computation:** it is a single standard scikit-learn
  model that fits in a few seconds.
- **Interpretability:** it is less interpretable than Logistic Regression
  and offers only feature importances. We accept this because the goal is
  the best F1. The tuned Logistic Regression would be the alternative if
  interpretability mattered more.

The final model is refitted on all 9,247 training sessions.
`predictions.csv` contains `ID` and `Revenue` for all 3,083 test sessions,
with predictions recorded as `True` or `False`.
