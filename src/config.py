"""Central configuration for the Portuguese bank marketing project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR    = PROJECT_ROOT / "dataset"
MODELS_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR    = PROJECT_ROOT / "logs"

for _d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "y"
N_SPLITS = 5

# THE most important line in this file.
#
# 'duration' is the length of the marketing call in seconds. You do not know it
# until the call has already happened, and by then you also know the answer. The
# dataset authors say so explicitly, and your brief repeats the warning. Include
# it and ROC-AUC jumps to roughly 0.93; exclude it and you get roughly 0.80.
# The 0.93 model is worthless in production because the feature cannot exist at
# prediction time.
#
# We build BOTH: the leaky one as a benchmark, clearly labelled, and the honest
# one for production. Reporting only the 0.93 is the single most common way this
# project goes wrong.
LEAKY_FEATURES = ["duration"]

# 999 in pdays does not mean "999 days ago". It is a sentinel meaning "never
# previously contacted". Left as a number it drags every split and coefficient.
PDAYS_SENTINEL = 999

EDUCATION_ORDER = ["illiterate", "basic.4y", "basic.6y", "basic.9y",
                   "high.school", "professional.course", "university.degree"]
MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
DAY_ORDER = ["mon", "tue", "wed", "thu", "fri"]

# Macro indicators. These move with the calendar, not with the customer, so they
# double as a proxy for TIME. That matters for how we validate: see src/split.py.
MACRO_FEATURES = ["emp.var.rate", "cons.price.idx", "cons.conf.idx",
                  "euribor3m", "nr.employed"]

# The bank controls these. Recommendations must come from here, because telling
# marketing to "improve the euribor rate" is not advice.
CONTROLLABLE_FEATURES = ["contact", "month", "day_of_week", "campaign"]

# Campaign economics for the profit curve. REPLACE with the bank's real numbers.
#
# These decide the answer, so state them explicitly rather than burying them.
# Watch the arithmetic: if value x base_rate > cost, calling EVERYONE is
# profitable and the model's job is prioritising a fixed call-centre capacity
# rather than deciding whom to skip. If value x base_rate < cost, there is a
# genuine interior optimum and the model decides where to stop. Both are real
# situations; which one you are in changes the recommendation entirely, so
# compute it before interpreting the curve.
COST_PER_CALL = 8.0
VALUE_PER_CONVERSION = 60.0
