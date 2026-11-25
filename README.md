# QuakeTensor

**Physics-informed seismic intelligence.** Understanding the Earth through physics, data and machine learning.

QuakeTensor makes prospective earthquake forecasts for California (and Italy, Japan, New Zealand) and scores them the way CSEP scores models. The goal is a physics-informed ML model that beats ETAS on forecasts committed before the outcome is known.

## Why ETAS is the bar

The Epidemic-Type Aftershock Sequence (ETAS) model is a statistical model built on two empirical laws of earthquake physics: the Omori–Utsu decay of aftershock rates in time, and Gutenberg–Richter magnitude scaling, with every event able to trigger its own aftershocks. It has a handful of parameters and has been the standard short-term forecasting model for decades.

**Almost no ML model beats it.** Deep-learning and neural point-process forecasters regularly look good in papers, but when they are compared with ETAS on the same catalogs, windows and likelihood scores, they mostly match it or fall short. The EarthquakeNPP benchmark (Stockman, Lawson and Werner, 2024) tested five neural point processes against ETAS on its benchmark catalogs and found none outperformed ETAS. The few reported wins are retrospective, on a single region, or not scored with CSEP's tests.

That is why QuakeTensor keeps ETAS as the baseline, puts the physics into the ML model rather than replacing it, and only counts a result if it shows positive information gain over ETAS on forecasts committed before the outcome.

## Quick start

Open the folder in VS Code and choose **Reopen in Container**. The container installs everything with `uv sync`. Then:

```bash
uv run quaketensor fetch --start 2018-01-01 --min-mag 3.0
uv run quaketensor forecast data/california_comcat.parquet --days 7
uv run quaketensor verify commitments/<file>.json
uv run quaketensor score commitments/<file>.json data/california_comcat.parquet
```

For exploratory analysis, fetch every ComCat column (ids, magnitude type, location quality, event type) and print the headline statistics:

```bash
uv run quaketensor fetch --full --start 1985-01-01 --min-mag 2.5
uv run quaketensor eda data/california_comcat_full.parquet --m0 3.0
```

`notebooks/02_eda.ipynb` is the full analysis: column descriptions, missing values, outliers, skewness, completeness and b-value, overdispersion, heteroskedasticity, seasonality, Omori decay and feature-target correlations. Its main finding is that declustered seismicity is Poisson with no seasonality, so the forecastable signal is the triggered (aftershock) part that ETAS already models.

`score` refuses to run until the catalog covers the whole window, so refetch after the window closes.

## Workflow for a verifiable result

1. Fit and commit a forecast before the window opens. The JSON in `commitments/` holds the forecast and its SHA-256 hash.
2. Commit that file to git and push, and post the hash publicly (LinkedIn, a tweet). Either gives you an independent timestamp.
3. After the window closes, refetch ComCat and run `score`. Anyone can rerun `verify` to confirm the forecast was not edited.
4. Report the N-test, Poisson log score and Brier score, and the information gain of your model over ETAS on the same windows.

## Layout

| Path | Purpose |
| --- | --- |
| `src/quaketensor/catalog.py` | ComCat download (pyCSEP, or full FDSN columns), column descriptions, regions, parquet I/O |
| `src/quaketensor/etas.py` | Temporal ETAS likelihood, MLE fit, b-value, forecasts |
| `src/quaketensor/evaluate.py` | N-test, Poisson log score, Brier score, information gain |
| `src/quaketensor/commit.py` | Hash commitments and verification |
| `src/quaketensor/eda.py` | EDA helpers: data quality, Mc and b-value, declustering, Omori fits, stationarity, heteroskedasticity, seasonality |
| `src/quaketensor/plots.py` | Shared plot style and EDA plots |
| `scripts/build_eda_notebook.py` | Generates `notebooks/02_eda.ipynb` |
| `src/quaketensor/cli.py` | `quaketensor` command |
| `notebooks/` | Executed exploration notebooks |
| `commitments/` | Tracked in git; `data/` and `forecasts/` are not |

## Baseline notes

ETAS v0 is temporal only and its forecast counts triggering from events before the window, not cascades inside it, so it underestimates rates right after a large shock. Next steps are a spatio-temporal ETAS with gridded output in pyCSEP format, then the ML model on top.

The CSEP ten-year California next-day forecast benchmark is on Zenodo; score against those models with pyCSEP's gridded tests once forecasts are gridded.

## Related work: MIT Lincoln Laboratory's QuakeCast

A team at MIT Lincoln Laboratory works on the same broad problem under the name QuakeCast (Reid, Liu and Ananthabhotla). Their approach is different from ours:

- **Data.** They use ionospheric total electron content (TEC) measured by GNSS receivers such as GPS, not the earthquake catalog. The idea is that the ionosphere may show anomalies hours to days before a large earthquake.
- **Method.** They treat it as classification. A neural network looks at a 24-hour sequence of TEC data in a regional window and labels it preseismic (an earthquake follows) or nominal.
- **Target.** They aim to warn of large earthquakes about eight hours to a week ahead. They report about 65% accuracy after switching from global to regional ionospheric data.

QuakeTensor works only from the seismicity catalog (USGS ComCat). It keeps the physics of earthquake triggering (Omori–Utsu, Gutenberg–Richter, ETAS) at the core. Its output is a probabilistic rate forecast rather than a yes/no alarm, and it is scored with CSEP's likelihood tests against ETAS. The two approaches could complement each other: precursor signals such as TEC could, in principle, enter a model like this one as an extra input.


## References

- Reid, J., Liu, J. and Ananthabhotla, B. *QuakeCast, an Earthquake Forecasting System Using Ionospheric Anomalies and Machine Learning.* ESS Open Archive, doi:10.1002/essoar.10506273.1
- MIT Lincoln Laboratory, *Predicting earthquakes by monitoring the ionosphere*: https://www.ll.mit.edu/news/predicting-earthquakes-monitoring-ionosphere
