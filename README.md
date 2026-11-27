# QuakeTensor

**Physics-informed seismic intelligence.** Understanding the Earth through physics, data and machine learning.

Prospective earthquake forecasts for California (plus Italy, Japan and New Zealand), scored with CSEP tests. The goal is a physics-informed ML model that beats ETAS on forecasts committed before the outcome is known.

## Why ETAS is the bar

ETAS builds Omori–Utsu aftershock decay and Gutenberg–Richter scaling into a few-parameter model and has been the short-term standard for decades. Neural forecasters rarely beat it on equal terms: EarthquakeNPP (Stockman, Lawson and Werner, 2024) found none of five did.
So ETAS stays the baseline, and a result only counts if it shows information gain over ETAS on pre-committed forecasts.

## Quick start

Open the folder in VS Code and choose **Reopen in Container** (it runs `uv sync`), then:

```bash
uv run quaketensor fetch --start 2018-01-01 --min-mag 3.0
uv run quaketensor forecast data/california_comcat.parquet --days 7
uv run quaketensor verify commitments/<file>.json
uv run quaketensor score commitments/<file>.json data/california_comcat.parquet
```

For EDA, fetch every ComCat column and print the headline statistics:

```bash
uv run quaketensor fetch --full --start 1985-01-01 --min-mag 2.5
uv run quaketensor eda data/california_comcat_full.parquet --m0 3.0
```

`notebooks/02_eda.ipynb` is the full analysis: declustered seismicity is Poisson with no seasonality, so the forecastable signal is the aftershock part ETAS models.
`score` only runs once the data cover the whole forecast window.

## Workflow for a verifiable result

Commit a hashed forecast to `commitments/` before the window opens, push it and post the hash publicly.
After the window closes, refetch ComCat, run `score`, and report the N-test, Poisson log score, Brier score and information gain over ETAS. Anyone can rerun `verify`.

## Layout

| Path | Purpose |
| --- | --- |
| `src/quaketensor/events.py` | ComCat download, column descriptions, regions, parquet I/O |
| `src/quaketensor/etas.py` | Temporal ETAS likelihood, fit, b-value, forecasts |
| `src/quaketensor/evaluate.py` | N-test, log score, Brier score, information gain |
| `src/quaketensor/commit.py` | Hash commitments and verification |
| `src/quaketensor/eda.py` | EDA helpers: data quality, Mc, b-value, declustering, Omori fits, seasonality |
| `src/quaketensor/plots.py` | Plot style and EDA plots |
| `src/quaketensor/cli.py` | `quaketensor` command |
| `scripts/build_eda_notebook.py` | Generates `notebooks/02_eda.ipynb` |
| `commitments/` | Tracked in git; `data/` and `forecasts/` are not |

## Baseline notes

ETAS v0 is temporal and ignores cascades inside the window, so it underestimates rates right after a large shock.
Next: spatio-temporal ETAS with gridded pyCSEP output, scored against the CSEP ten-year California next-day benchmark, then the ML model.

## Related work

MIT Lincoln Laboratory's QuakeCast (Reid, Liu and Ananthabhotla) classifies 24-hour windows of GNSS ionospheric TEC as preseismic or nominal, aiming for warnings 8 hours to a week ahead (about 65% accuracy regionally).
QuakeTensor instead forecasts rates from ComCat seismicity with ETAS physics at the core, scored with CSEP tests; TEC could later be an extra input.

## References

- Reid, J., Liu, J. and Ananthabhotla, B. *QuakeCast, an Earthquake Forecasting System Using Ionospheric Anomalies and Machine Learning.* ESS Open Archive, doi:10.1002/essoar.10506273.1
- MIT Lincoln Laboratory, *Predicting earthquakes by monitoring the ionosphere*: https://www.ll.mit.edu/news/predicting-earthquakes-monitoring-ionosphere
