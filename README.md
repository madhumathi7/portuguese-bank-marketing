# Portuguese Bank Direct Marketing (PRCP-1000)

Predicting which customers subscribe to a term deposit, from 41,188 phone-call
campaign records collected May 2008 to November 2010.

## The headline finding

**`duration` must be excluded, and doing so costs about 0.30 ROC-AUC.**

Call duration is the strongest correlate of the outcome in the raw data, reaching
a univariate AUC above 0.90 on its own. It is also unusable: you do not know how
long a call lasted until it has ended, and by then you already know whether the
customer said yes. The long call is a *consequence* of interest, not a predictor
of it. The dataset authors say this explicitly and the project brief repeats it.

Both models are reported here. The one with `duration` is a labelled benchmark
that exists only to quantify the gap. The one without is the production model.
Reporting only the higher number would mean shipping something that predicts
nothing.

| Model | ROC-AUC (5-fold) | ROC-AUC (temporal) | Lift @ top 10% | Fit time |
|---|---|---|---|---|
| With duration — benchmark only | [fill from reports/] | | | |
| Logistic Regression | | | | |
| Decision Tree | | | | |
| Random Forest | | | | |
| LightGBM | | | | |
| XGBoost | | | | |

**Recommended for production:** [fill in]

Full analysis: `reports/model_comparison_report.md` and `reports/challenges_report.md`

## Other decisions that shaped the result

**`pdays = 999` is a sentinel, not a number.** It means "never previously
contacted" and covers about 86% of rows. Left as a magnitude, a linear model fits
a slope through a cliff and a tree spends its first split rediscovering that 999
is special. Split into a binary contacted-before flag plus real recency.

**`unknown` is a category, not missing data.** It appears in six columns across
roughly 12,800 records. There are no true NaNs in this file. A customer declining
to state their job or education is itself informative, so `unknown` is kept as a
level rather than dropped or imputed.

**The macro indicators encode time.** `euribor3m`, `nr.employed` and
`emp.var.rate` move with the calendar, not the customer, and this campaign ran
straight through the 2008 financial crisis. Under a random split they let the
model recognise *when* a row is from, which is a mild form of leakage. Both a
random 5-fold score and a temporal holdout score are reported; the temporal one
answers the question the bank is actually buying — how will this work next
quarter.

## Task 3 — recommendations for the marketing team

Every suggestion comes from a feature the bank *controls*. Age, job and the
euribor rate are strong predictors and useless as advice: "target younger
customers when rates are low" is an observation, not something the campaign team
can act on next Monday. Contact channel, month, day of week and contact frequency
are all decisions someone makes.

See `reports/recommendations.md`. Each recommendation carries its conversion rate,
a 95% Wilson interval, and a confidence flag — because a 40% success rate on 12
customers is noise, and without an interval it ends up in a slide deck as strategy.

The final recommendation is to run a randomised holdout on the next campaign.
Everything here is observational and shows association; only an experiment shows
that changing a lever causes the lift.

## Install

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install --upgrade pip
pip install -r requirements-dev.txt
python -m ipykernel install --user --name pbank --display-name "Python (PBank)"
```

Place `bank-additional-full.csv` inside `dataset/`. Note it is **semicolon
separated** — reading it with the default comma separator yields a single-column
frame and no error.

## Usage

```bash
jupyter lab                      # run notebooks/PBank_Complete.ipynb
python -m pytest tests/ -v
python app.py                    # optional call-list service on :5000
```

## Project structure

```
src/config.py       paths, seed, the leaky-feature list, campaign economics
src/data_loader.py  loading, audit, quantified evidence for the duration leak
src/features.py     pdays sentinel fix, derived features, preprocessing pipeline
src/split.py        stratified folds and the time-aware holdout
src/models.py       six candidates, each testing a stated hypothesis
src/train.py        CV, temporal evaluation, profit curve, threshold selection
src/recommend.py    Task 3, restricted to controllable levers
src/plots.py        every figure
src/persist.py      save and load the full pipeline plus threshold
app.py              optional call-list scoring service
```

## Known limitations

- Observational data with no experimental variation, so no recommendation here is
  proven causal.
- The campaign ran through the 2008 financial crisis. Macro conditions since then
  differ, and the macro coefficients are unlikely to transfer.
- Campaign economics in `config.py` are assumptions. Replace `COST_PER_CALL` and
  `VALUE_PER_CONVERSION` with the bank's real figures before quoting any profit
  number.
- No customer identifier, so repeat customers across campaigns cannot be tracked.

## License

MIT. See `LICENSE`.
