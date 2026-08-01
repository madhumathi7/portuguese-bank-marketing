# Model comparison report - PRCP-1000 Portuguese Bank

## Protocol
Stratified 5-fold cross-validation over 11,162 records
(47.38% subscribe), seed 42, plus a temporal holdout training on
the earlier 80% of the campaign and testing on the later 20%.

**Accuracy is not reported.** Predicting "no" for everyone scores
52.62% with zero skill. ROC-AUC is primary, PR-AUC is reported under
imbalance, and lift at the top 10% translates the result into campaign terms.

## The duration exclusion
`duration` reaches a univariate ROC-AUC of 0.8184
on its own, and successful calls run 2.5x longer than unsuccessful
ones. It is excluded from every production model because it is not known until the
call has ended, at which point the outcome is known too.

Measured cost of that exclusion, same model and same folds:

| Feature set | ROC-AUC | Lift @ 10% |
|---|---|---|
| With duration (benchmark only) | 0.9252 | 1.98x |
| Without duration (production) | 0.7863 | 1.94x |

The +0.1389 difference is hindsight,
not skill.

## Results

| model              |   oof_roc_auc |   oof_pr_auc |   oof_lift_at_10 |   temporal_roc_auc |   gap |   fit_seconds_total |
|:-------------------|--------------:|-------------:|-----------------:|-------------------:|------:|--------------------:|
| XGBoost            |        0.7901 |       0.7897 |           1.9345 |                nan |   nan |             11.1936 |
| LightGBM_balanced  |        0.7859 |       0.7838 |           1.9062 |                nan |   nan |              6.9091 |
| LightGBM           |        0.7859 |       0.7835 |           1.9119 |                nan |   nan |              6.7808 |
| RandomForest       |        0.7793 |       0.7739 |           1.91   |                nan |   nan |             23.4103 |
| LogisticRegression |        0.7636 |       0.7628 |           1.9062 |                nan |   nan |              2.9641 |
| DecisionTree       |        0.7258 |       0.7131 |           1.9232 |                nan |   nan |              1.8545 |

## Random CV versus temporal holdout
Mean optimism from random cross-validation: **+nan AUC**. The
macro indicators move with the calendar, so a random split lets the model
recognise which period a row belongs to, and rows from a period share an outcome
rate. The temporal figure is the one to quote as expected production performance.

## Recommendation for production
**XGBoost**: cross-validated ROC-AUC 0.7901
(+/- 0.0164), temporal holdout
nan, lift 1.93x on the
top decile, fitting in 11s across all folds.

Where two models sit within one fold-standard-deviation of each other, prefer the
simpler and more interpretable one. Task 3 requires explaining the model to a
marketing team, and an odds ratio can go in a slide where a boosted ensemble's
400th tree cannot.

## Operating point
Threshold 0.0969, chosen from the profit curve rather than left at 0.5.
At 0.5 the model would flag only 4,417 customers, because a
calibrated model rarely scores anyone above 0.5 when the base rate is
47.4%. Optimum campaign size is
97% of the list, giving precision
0.485 and recall
0.992.

Campaign economics used: 8.00 per call,
60.00 per conversion. **These are assumptions.**
Replace them with the bank's real figures before quoting any profit number.

## Caveats
- Fold-to-fold standard deviation is 0.0164. Treat smaller
  differences between models as noise.
- The campaign ran through the 2008 financial crisis; the macro coefficients are
  unlikely to transfer to a different rate environment.
