# credit-liquidity-monitor
A real-time credit and liquidity stress monitor that combines market-based and funding signals to detect early signs of financial instability.

## Composite indicator

`stress_model.py` calculates the production score and the notebook calls the same function. The inputs are rolling 60-observation z-scores for high-yield OAS, BBB OAS, their difference, SOFR minus EFFR, reverse-repo usage, and optionally repo usage. Their respective weights are 30%, 20%, 5%, 35%, 5%, and 5%. Available weights are renormalized when an optional facility series is missing. High HY, BBB, and SOFR minus EFFR levels provide a floor for those components so persistent stress does not disappear as the rolling mean catches up.

HY OAS, BBB OAS, SOFR, and EFFR are required. A required observation older than three business days, or an insufficient rolling baseline, makes the current score unavailable and the regime `UNKNOWN`. The email includes data coverage. An unavailable score raises a data-quality alert.

The displayed **Stress Index** is `100 / (1 + exp(-score))`. It is an uncalibrated display scale, **not a probability**. Heuristic regimes begin at index 60 (WATCH), 75 (STRESS), and 85 (CRISIS). Treat these cutoffs as operational alerts until longer-history, out-of-sample validation supports them.

The notebook at `notebooks/credit_liquidity_stress_backtest.ipynb` tests whether the production index is associated with subsequent BKLN, HYG, and SPY losses. It uses a later-period holdout with a gap equal to the forward-return horizon. Its optional PyMC model is exploratory. No historical probability calibration is implemented because the repo has no agreed stress-event definition or labeled stress dataset.

Run the offline indicator checks with `poetry run python -m unittest discover -s tests -v`.
