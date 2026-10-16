# Machine learning methodology

Seed: 42. The evaluation below is the run that wrote `ml/artifacts/metadata.json` on the IBM Telco Customer Churn file (7,043 trainable rows).

## Split

The labeled table is split before any imputer, encoder, scaler, or model is fit.

- 20% stratified holdout, locked until the model, calibration, and threshold are frozen.
- The remaining 80% is split 75/25 into train and internal validation.
- Resulting sizes: 4,225 train, 1,409 validation, 1,409 test. The three index sets are disjoint.
- Churn rate in each split is about 26.5%.

Preprocessing lives inside a scikit-learn `Pipeline` and `ColumnTransformer`. Numeric columns use median imputation. Categorical columns use most-frequent imputation and one-hot encoding with unknown categories ignored. Scaling is applied only for logistic regression. The test rows are never passed to `fit`.

`gender` and `SeniorCitizen` are kept for descriptive charts and are excluded from the model, explanations, clusters, and recommendations.

## Models compared

Selection uses mean cross-validated average precision on the training split (3 stratified folds). Accuracy, precision, recall, and F1 in the comparison table use the library cutoff of 0.5 and are not the selection metric. Ties would break on recall at 0.5, then ROC-AUC.

| Model | CV average precision | CV ROC-AUC | Recall at 0.5 | Precision at 0.5 | F1 at 0.5 | Accuracy at 0.5 | Fit seconds | Chosen parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic regression | 0.6696 | 0.8482 | 0.5558 | 0.6480 | 0.5983 | 0.8021 | 24.0 | C=1, no class weight |
| Decision tree | 0.6205 | 0.8292 | 0.7975 | 0.5071 | 0.6198 | 0.7401 | 15.5 | depth 8, min leaf 40, balanced |
| Random forest | 0.6736 | 0.8463 | 0.7574 | 0.5412 | 0.6310 | 0.7647 | 172.5 | 180 trees, depth 16, min leaf 8, balanced subsample |
| HistGradientBoosting | 0.6693 | 0.8442 | 0.7993 | 0.5224 | 0.6318 | 0.7527 | 1527.8 | learning rate 0.05, depth 3, min leaf 20, balanced |
| XGBoost | 0.6724 | 0.8462 | 0.7957 | 0.5198 | 0.6288 | 0.7505 | 276.1 | depth 3, learning rate 0.05, scale_pos_weight 2.77 |

Random forest is selected because it has the highest training-split average precision. The holdout was not used for that choice. HistGradientBoosting is slow in this configuration because early stopping is off and each candidate runs 180 iterations.

## Calibration, resampling, and threshold

- Random oversampling inside logistic-regression folds scored 0.6660 average precision. Class-weighted logistic regression scored 0.6672. Oversampling was not deployed, and logistic regression was not the selected model.
- Sigmoid `CalibratedClassifierCV` lowered validation Brier from 0.1569 to 0.1376, which clears the 0.001 margin, so calibration is part of the saved pipeline.
- The positive-class threshold minimizes `5 × false negatives + 1 × false positives` on the validation split. That 5:1 ratio is an assumption, not a measured cost. The chosen high-risk threshold is 0.16 (validation cost 624, validation precision 0.464, validation recall 0.866).
- Medium risk starts at 0.10, the lowest cutoff below 0.16 whose validation precision still meets the training churn rate of 0.265. Medium is an elevated band, not a second positive class.
- Validation metrics at this cutoff are optimistic because the cutoff was chosen on that split. Use the test metrics.

## Held-out test

The threshold search reads validation labels and validation probabilities only. Test labels are first used after that search, when the frozen threshold is applied to the refit model. The saved pipeline is refit on the combined train and validation rows before that one test evaluation, so the reported test metrics describe the refit model. The threshold was not searched again on the refit probabilities or on the test labels. Validation metrics at 0.16 are optimistic for that reason.

Evaluated once, after selection, calibration, and the threshold were frozen.

| Metric | Value |
| --- | ---: |
| Average precision | 0.6521 |
| ROC-AUC | 0.8441 |
| Precision at 0.16 | 0.4687 |
| Recall at 0.16 | 0.8610 |
| F1 at 0.16 | 0.6070 |
| Accuracy at 0.16 | 0.7040 |
| Brier score | 0.1366 |
| True retained / false churn | 670 / 365 |
| Missed churn / caught churn | 52 / 322 |

The low threshold trades precision for recall under the stated cost assumption. Of the 687 test customers scored at or above 0.16, 365 were labeled retained, so 53.1% of the positive flags are false positives. Of the 374 test customers labeled churned, 52 fall below 0.16 and are false negatives. A classifier that never predicts churn would be about 73.5% accurate on this split. The held-out accuracy of 0.7040 is lower than that majority baseline, which is expected for a cutoff chosen to catch churn rather than to maximize accuracy. Settings can raise the threshold without retraining.

## What this evaluation does not establish

The holdout is separated before imputation, encoding, model selection, calibration, and the threshold search. Preprocessing sits inside the pipeline, so each `fit` sees only the rows passed to it: training folds during search, then the combined train and validation rows for the saved model. Test labels are not used to choose the model, the calibration method, or the 0.16 threshold.

That is not a leakage-free evaluation. The calibration decision and the threshold both use internal-validation labels, so validation metrics at 0.16 are optimistic. The saved model is refit on train plus validation before the one test call, and there is no second untouched split to confirm that result. The split is a random stratified sample of one historical snapshot. The file has no event date, so the split is not temporal and does not estimate performance on a later month. Customers who fell in the test split are still scored in the application; those rows are operational displays of the same model, not an extra evaluation.

`TotalCharges` is largely determined by tenure and monthly charges. That is redundancy among predictors, not use of the churn label. `gender` and `SeniorCitizen` are excluded from the model, explanations, clusters, and recommendations. They remain available for descriptive charts. Permutation importance is measured on the validation split after selection. It did not choose the model and it is not a causal ranking.

Sigmoid calibration was kept because it lowered validation Brier from 0.1569 to 0.1376. The held-out Brier is 0.1366. The model page draws an 8-bin quantile reliability diagram from those held-out probabilities. That diagram was not used to choose calibration or the threshold. It is one split of one snapshot, so it does not show that the probabilities will match observed frequencies in a later period.

The positive class is about 26.5% of each split. The selected forest uses `balanced_subsample`. Oversampling was tried only on logistic regression and was not deployed. One seed (42), one cost ratio (5:1), and one dataset produced these numbers. They describe this artifact. They do not transfer to another company or a later period without a new evaluation.

Grouped permutation importance on the validation split, measured as the drop in average precision, ranks tenure, contract, and internet service first. It was not used to choose the model and it is not a causal ranking.

## Explanations

For one customer the service tries SHAP and keeps it only when the explanation reconciles with the predicted probability within 0.08. Otherwise it uses baseline occlusion: each feature is replaced with its training baseline and the change in predicted probability is recorded. Batch files use occlusion on the top permutation features. No SHAP value is invented.

The saved model is a sigmoid-calibrated pipeline. That wrapper is not a SHAP-supported estimator, so customer and batch explanations for this artifact use baseline occlusion. SHAP still runs for a directly supported estimator, such as an uncalibrated forest or logistic regression, when the values reconcile.

## Segments

Rule segments can overlap. Primary assignment uses a fixed priority order. K-means uses tenure, monthly charges, total charges, add-on count, month-to-month, fiber, and electronic check. It is fit on the training split only. Churn, gender, and senior-citizen status are not cluster features. Silhouette on the training matrix selected k=6 (0.386) over k=2 through k=5. Cluster names describe the fitted profiles.
