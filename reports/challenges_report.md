# Report on challenges faced - PRCP-1000 Portuguese Bank

## 1. Target leakage through `duration`
**Problem.** The strongest feature in the dataset cannot exist at prediction time.
Call duration is only known after the call, by which point the outcome is known.
**Technique.** Excluded from all production models; retained as an explicitly
labelled benchmark to quantify the gap. Evidence recorded rather than asserted:
univariate AUC 0.8184, successful calls
2.5x longer, zero-second calls never converting.
**Reason.** A model scoring 0.925 that cannot be deployed is
worth less than one scoring 0.786 that can. `src/persist.py`
strips the column at scoring time so it cannot be reintroduced by accident.

## 2. `pdays = 999` is a sentinel, not a magnitude
**Problem.** 999 means "never previously contacted" and covers
96.3% of rows. Treated as a number,
a linear model fits a slope through a cliff and a tree spends its first split
rediscovering that 999 is special.
**Technique.** Split into `was_contacted_before` (binary) and `pdays_recency`,
with the sentinel filled at the median of genuine values.
**Reason.** Encode what the data means. Two honest columns beat one misleading one.

## 3. `unknown` is a category, not missing data
**Problem.** Zero true NaNs, but the literal string 'unknown' appears in six
columns across 12,718 records.
**Technique.** Kept as its own level throughout, with
`OneHotEncoder(handle_unknown='ignore')` so unseen categories at serving time
become all-zeros rather than raising.
**Reason.** Non-disclosure is itself informative. Dropping those rows discards a
large fraction of the data to fix a problem that does not exist.

## 4. Macro indicators encode time, so random CV leaks
**Problem.** euribor3m, nr.employed and emp.var.rate move with the calendar, not
the customer. This campaign spans the 2008 crisis, so those columns let a model
identify the period a row came from - and rows from a period share an outcome rate.
**Technique.** Verified the file's chronological ordering empirically via Spearman
correlation between row index and each macro indicator, then reported both a random
5-fold score and a temporal holdout. Measured optimism: +nan AUC.
**Reason.** The two numbers answer different questions. The bank is buying "will
this work next quarter", which only the temporal split estimates.

## 5. Class imbalance at 47.4%
**Problem.** Accuracy is actively misleading and a 0.5 threshold flags almost no one.
**Technique.** Stratified folds throughout; ROC-AUC and PR-AUC as metrics; threshold
selected from the profit curve; `scale_pos_weight` tested as an explicit variant
rather than applied by default.
**Reason.** Under imbalance, accuracy measures the base rate, and 0.5 is a default
nobody chose.

## 6. Multicollinearity among the economic indicators
**Problem.** The five macro columns are near-duplicates, inflating variance in
linear coefficients and splitting importance across correlated features in trees.
**Technique.** Documented the correlation structure; used permutation importance
rather than impurity importance for interpretation.
**Reason.** Impurity importance is biased toward high-cardinality features and would
overstate the one-hot expanded columns. Permutation measures the effect on the
metric you actually report.

## 7. Turning predictions into advice (Task 3)
**Problem.** The most predictive features are the least actionable. Telling
marketing to "target customers when the euribor rate falls" is not a
recommendation.
**Technique.** Partitioned features into controllable and uncontrollable, and
sourced every recommendation from the controllable set. Attached Wilson confidence
intervals to each conversion rate.
**Reason.** Advice must map to a decision someone can make. Intervals prevent a
small-sample artefact becoming campaign policy.

## 8. Observational data cannot establish causality
**Problem.** Every recommendation is an association. Customers contacted by mobile
may differ systematically from those contacted by landline in ways the data does
not record.
**Technique.** Stated the limitation explicitly and recommended a randomised
holdout group on the next campaign.
**Reason.** It is the only way to convert these associations into causal evidence,
and proposing it is more useful than overclaiming.

## Future improvements
- Randomised control group in the next campaign to test the recommendations.
- Real cost and value figures to replace the assumed campaign economics.
- Isotonic calibration if predicted probabilities feed a revenue calculation.
- A customer identifier, which would allow tracking across campaigns and modelling
  contact fatigue at the person level rather than the record level.
- Drift monitoring on the macro features, which are the ones most likely to move.
