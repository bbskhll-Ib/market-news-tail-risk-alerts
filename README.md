# Market–News Models for Equity Tail-Risk Alerts

Code and processed result tables supporting the manuscript:

> Market–News Models for Equity Tail-Risk Alerts: Conservative Headline
> Timing, Calibration, and Evidence from the S&P 500 and NASDAQ

## Overview

This repository provides the Python pipeline and processed output tables for
an end-of-day equity tail-risk alert system. The model uses market features
observed through the close of session t and headline-derived features lagged
by one trading session to predict a five-session forward-volatility event over
sessions t+1 through t+5.

The study evaluates logistic regression and LightGBM models using
chronologically separated training (2008–2016), calibration (2017–2018), and
held-out test (2019–2024) periods.

## Data access

Market data are retrieved through `yfinance` for:

- `^GSPC` — S&P 500
- `^IXIC` — NASDAQ Composite
- `^VIX` — CBOE Volatility Index

Headline data must be obtained directly from the original Kaggle dataset:

- [Mahaptra, D.D. (2024). *S&P 500 with Financial News Headlines (2008–2024)*]
(https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024)*

The raw headline data and headline-level FinBERT outputs are not included in
this repository because they are subject to the original data provider's
terms.

After obtaining the raw file, place it in the repository root as:

```text
sp500_headlines_2008_2024.csv
```

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Reproduction

Run:

```bash
python code/tail_risk_pipeline.py
```

The script downloads market data, processes headlines with FinBERT, fits the
models, calibrates probabilities, calculates metrics, and writes output files
to `outputs_best_corrected/`.

## Timing design

The prediction is generated immediately after the close of trading session t.

- Market features use information observed through the close of session t.
- Date-only headlines are assigned to the first trading session on or after
  their recorded date and lagged by one trading session.
- The target uses returns from sessions t+1 through t+5.

## Main configuration

- Forward horizon: 5 trading sessions
- Event threshold: 85th percentile of training-inner forward volatility
- Bootstrap: 800 circular moving-block resamples
- Block length: 20 trading sessions
- Calibration: sigmoid calibration fitted on 2017–2018
- Operating point: calibration-window FPR ≤ 5%

## Repository contents

- `code/`: analysis pipeline
- `outputs/`: processed result tables reported in the manuscript
- `figures/`: figure files, if included

## Reproducibility note

Historical vendor data can change over time. The processed output tables in
`outputs/` are included to document the numerical results reported in the
manuscript.

## License

The code is released under the MIT License. The raw headline data are not
redistributed.
