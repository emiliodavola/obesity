# %% [markdown]
# # Audit — 01_baseline: baseline classification with TabPFN
#
# Read-only review of the stored report below: its checks and metrics.

# %%
import skore
import os

# %% [markdown]
# ## Open the project
#
# Open the same project the experiment wrote to. The init block below
# must match `experiments/01_baseline.py` exactly — copy it from
# there rather than retyping it.

# %%
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), 'experiments'))
project = skore.Project(name="obesity", workspace=os.path.abspath(os.path.join(os.getcwd(), 'experiments')), mode="local")
project

# %% [markdown]
# ## List the available reports
#
# `project.summarize()` provides an overview of all reports in this
# Project — useful to confirm the experiment's report landed and to
# spot duplicate keys from accidental re-runs.

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report
#
# **Hub mode** — `project.put()` prints a URL of the form:
#
#   `https://skore.probabl.ai/<hub-workspace>/<project>/<type-plural>/<N>`
#
# The report id is `skore:report:<type-singular>:<N>`.  The URL path
# segment is the plural; the id uses the singular (drop the trailing
# `s`).  Examples:
#
#   `.../cross-validations/42`  ->  `skore:report:cross-validation:42`
#   `.../estimators/7`          ->  `skore:report:estimator:7`
#
# **Local mode** — `project.put()` does not print a URL.  Read the
# `"id"` column value from the `summary` DataFrame above for the row
# whose `"key"` matches this experiment's stem, and set `REPORT_ID` to
# that value.

# %%
REPORT_ID = "195398642518948936502400246745309599323"

report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Checks summary
#
# `report.checks.summarize().frame()` returns a DataFrame whose rows
# each carry a `code` (e.g. `SKD003`), a `severity` (`passed` /
# `issue` / `tip`), and a `documentation_url` — the linked page
# describes what the check tests and what to try next.
#
# Available on `EstimatorReport` and `CrossValidationReport` in
# skore >= 0.18. Mute a noisy check via
# `report.checks.summarize(ignore=['<code>']).frame()`.
#
# Docs: https://docs.skore.probabl.ai/0.18/user_guide/automated_checks.html

# %%
report.checks.summarize().frame()

# %% [markdown]
# ## Metrics summary
#
# `report.metrics.summarize().frame()` covers task-appropriate
# defaults in one call:
#
# - regression: RMSE / MAE / R² + fit/predict timings,
# - binary classification: accuracy / precision / recall / F1 /
#   ROC-AUC / log-loss + timings,
# - multiclass: macro / micro averages of the above.
#
# Same accessor on both `EstimatorReport` and
# `CrossValidationReport`; the latter additionally reports mean ±
# std across folds. This is the headline reading; the actionable
# findings come from the checks section above.

# %%
report.metrics.summarize().frame()

# %% [markdown]
# ## End of audit
#
# This file is the durable record of how the experiment's report was
# reviewed; re-run it any time to refresh the checks and metrics
# above.
